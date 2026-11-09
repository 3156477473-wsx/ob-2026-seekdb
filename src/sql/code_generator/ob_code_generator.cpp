/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_CG

#include "config_bridge.h"
#include "sql/code_generator/ob_code_generator.h"
#include "sql/code_generator/ob_static_engine_cg.h"
#include "sql/code_generator/ob_dml_cg_service.h"
#include "sql/engine/ob_physical_plan.h"
#include "sql/optimizer/ob_log_table_scan.h"
#include "share/schema/ob_schema_getter_guard.h"

namespace oceanbase
{
namespace sql
{
namespace
{
// This is a term-expanding online-DDL scan, not a generally small document scan.
// The runtime/DAS scalar-source checks in generate_spec_final remain unchanged.
bool can_expand_fts_term_batch(const ObLogPlan &plan, uint64_t &ddl_table_id)
{
  ObOptimizerContext &opt_ctx = plan.get_optimizer_context();
  ObExecContext *exec_ctx = opt_ctx.get_exec_ctx();
  if (!opt_ctx.is_insert_stmt_in_online_ddl() || nullptr == exec_ctx
      || nullptr == exec_ctx->get_sql_ctx() || nullptr == exec_ctx->get_sql_ctx()->schema_guard_) {
    return false;
  }
  const TableItem *target = opt_ctx.get_root_stmt()->get_table_item(0);
  const share::schema::ObTableSchema *schema = nullptr;
  if (nullptr == target || common::OB_INVALID_ID == target->ddl_table_id_
      || OB_SUCCESS != exec_ctx->get_sql_ctx()->schema_guard_->get_table_schema(target->ddl_table_id_, schema)
      || nullptr == schema || (!schema->is_fts_index_aux() && !schema->is_fts_doc_word_aux())
      || schema->get_parser_name_str().empty()) {
    return false;
  }
  // Require one filter-free unary path to one whole-range base scan. Joins,
  // lookups, parameterized scans and other expanding index types fall back.
  ObLogicalOperator *node = plan.get_plan_root();
  int64_t depth = 0;
  while (nullptr != node && !node->is_table_scan() && depth++ < 64) {
    if (1 != node->get_num_of_child() || !node->get_filter_exprs().empty()
        || !node->get_startup_exprs().empty()) {
      return false;
    }
    node = node->get_child(0);
  }
  if (nullptr == node || !node->is_table_scan() || 0 != node->get_num_of_child()
      || !node->get_filter_exprs().empty() || !node->get_startup_exprs().empty()) {
    return false;
  }
  ObLogTableScan *scan = static_cast<ObLogTableScan *>(node);
  if (scan->get_ref_table_id() != schema->get_data_table_id()
      || !scan->is_table_whole_range_scan() || scan->get_index_back() || scan->use_batch()
      || scan->use_index_merge() || scan->is_scan_resumable() || scan->has_func_lookup()
      || scan->get_is_spatial_index() || scan->is_spiv_vec_scan()
      || nullptr != scan->get_limit_expr() || nullptr != scan->get_offset_expr()) {
    return false;
  }
  const auto &domains = scan->get_rowkey_domain_types();
  if (domains.count() > 1 || (1 == domains.count()
      && share::ObDomainIdUtils::ObDomainIDType::DOC_ID != domains.at(0))) {
    return false;
  }
  const auto &outputs = scan->get_output_exprs();
  if (4 != outputs.count() && 5 != outputs.count()) {
    return false;
  }
  int64_t word_idx = -1;
  int64_t counts[3] = {0, 0, 0};
  int64_t partitions = 0;
  for (int64_t i = 0; i < outputs.count(); ++i) {
    const ObRawExpr *expr = outputs.at(i);
    if (nullptr == expr || !expr->is_vectorize_result()) {
      return false;
    } else if (T_FUN_SYS_WORD_SEGMENT == expr->get_expr_type()) {
      word_idx = i;
      ++counts[0];
    } else if (T_FUN_SYS_WORD_COUNT == expr->get_expr_type()) {
      ++counts[1];
    } else if (T_FUN_SYS_DOC_LENGTH == expr->get_expr_type()) {
      ++counts[2];
    } else if (T_PDML_PARTITION_ID == expr->get_expr_type()) {
      ++partitions;
    }
  }
  const int64_t doc_idx = word_idx + (schema->is_fts_index_aux() ? 1 : -1);
  if (1 != counts[0] || 1 != counts[1] || 1 != counts[2]
      || partitions != outputs.count() - 4 || doc_idx < 0 || doc_idx >= outputs.count()
      || T_REF_COLUMN != outputs.at(doc_idx)->get_expr_type()) {
    return false;
  }
  ddl_table_id = target->ddl_table_id_;
  return true;
}

bool fts_expr_frames_fit(const ObRawExprUniqueSet &exprs, const int64_t batch_size,
                        const int64_t budget, const int64_t lob_max_rows,
                        const int64_t parameter_count, const int64_t original_parameter_count,
                        int64_t &bound)
{
  bound = 0;
  const auto decline = [&](const char *reason, const int64_t index, const ObRawExpr *expr) {
    const int64_t expr_count = exprs.count();
    const int64_t expr_kind = nullptr == expr ? -1 : expr->get_expr_type();
    const int64_t datum_type = nullptr == expr ? -1 : expr->get_data_type();
    const bool vector_result = nullptr != expr && expr->is_vectorize_result();
    const bool static_const = nullptr != expr && expr->is_static_const_expr();
    const bool attribute_expr = nullptr != expr && expr->need_attr_expr();
    const bool dynamic_questionmark = nullptr != expr && T_QUESTIONMARK == expr->get_expr_type()
        && static_cast<const ObConstRawExpr *>(expr)->is_dynamic_eval_questionmark();
    LOG_INFO("FTS term expansion frame declined", K(reason), K(index), K(batch_size), K(bound), K(expr_count));
    LOG_INFO("FTS term expansion declined expression", K(expr_kind), K(datum_type),
             K(vector_result), K(static_const), K(dynamic_questionmark), K(attribute_expr));
    return false;
  };
  if (batch_size <= 0 || budget <= 0) {
    return decline("invalid_budget", -1, nullptr);
  }
  if (parameter_count < 0 || original_parameter_count < 0) {
    return decline("unknown_parameter_frames", -1, nullptr);
  }
  // Parameter frames cover the whole store, including unused slots. Runtime
  // rounds the extra segment up to a power of two: twice max(P,O) bounds both
  // original and rounded extra capacity. Visible QMs are also charged below.
  const int64_t max_parameters = parameter_count > original_parameter_count
      ? parameter_count : original_parameter_count;
  const int64_t parameter_slot_bound = 2 * (sizeof(ObDatum) + sizeof(ObEvalInfo));
  if (max_parameters > budget / parameter_slot_bound) {
    return decline("parameter_frame_capacity", -1, nullptr);
  }
  bound = max_parameters * parameter_slot_bound;
  LOG_INFO("FTS term expansion parameter frames", K(batch_size), K(parameter_count),
           K(original_parameter_count), K(bound), K(budget));
  for (int64_t i = 0; i < exprs.count(); ++i) {
    const ObRawExpr *expr = exprs.get_expr_array().at(i);
    // Extra dynamic questionmarks are introduced after this preflight. Do not
    // promote a plan whose complete frame set cannot be bounded here.
    if (nullptr == expr) {
      return decline("null_expression", i, expr);
    } else if (T_QUESTIONMARK == expr->get_expr_type()
        && static_cast<const ObConstRawExpr *>(expr)->is_dynamic_eval_questionmark()) {
      return decline("dynamic_questionmark", i, expr);
    } else if (T_OP_ROW == expr->get_expr_type()) {
      return decline("row_expression", i, expr);
    } else if (expr->need_attr_expr()) {
      return decline("generated_attributes", i, expr);
    }
    const common::ObObjType type = expr->get_data_type();
    if (type < common::ObNullType || type >= common::ObMaxType) {
      return decline("invalid_datum_type", i, expr);
    }
    const bool large_data = common::ObLongTextType == type || common::ObMediumTextType == type
        || common::ObTextType == type || common::ObJsonType == type
        || common::ObUserDefinedSQLType == type || common::ObCollectionSQLType == type;
    if (expr->is_vectorize_result() && large_data && batch_size > lob_max_rows) {
      return decline("lob_row_limit", i, expr);
    }
    const common::ObObjDatumMapType map_type = ObDatum::get_obj_datum_map_type(type);
    if (map_type >= common::OBJ_DATUM_MAPPING_MAX) {
      return decline("unmapped_datum_type", i, expr);
    }
    const int64_t n = expr->is_vectorize_result() ? batch_size : 1;
    // Use the uncapped default reserve: runtime max_length can only reduce it.
    // Include scalar/constant expressions too, conservatively, rather than
    // counting only the four output fields or the detector's filtered subset.
    const int64_t reserve = ObDatum::get_reserved_size(map_type, expr->get_result_type().get_precision());
    const int64_t per_row = sizeof(ObDatum) + reserve
        + (ObDynReserveBuf::supported(type) ? sizeof(ObDynReserveBuf) : 0);
    if (per_row > budget / n) {
      return decline("per_row_budget", i, expr);
    }
    const int64_t expr_bytes = n * per_row + sizeof(ObEvalInfo)
        + 2 * ObBitVector::memory_size(n);
    if (expr_bytes > common::MAX_FRAME_SIZE || expr_bytes > budget - bound) {
      LOG_INFO("FTS term expansion frame size declined", K(i), K(n), K(per_row), K(expr_bytes), K(budget), K(bound));
      return decline(expr_bytes > common::MAX_FRAME_SIZE ? "frame_limit" : "whole_frame_budget", i, expr);
    }
    bound += expr_bytes;
  }
  return true;
}
} // namespace

int ObCodeGenerator::generate(const ObLogPlan &log_plan,
                              ObPhysicalPlan &phy_plan)
{
  int ret = OB_SUCCESS;
  int64_t batch_size = 0;
  bool has_ai_columns = false;
  OZ(ObDmlCgService::prepare_ai_expressions(*log_plan.get_plan_root(), has_ai_columns));
  if (OB_SUCC(ret) && !has_ai_columns) {
    OZ(detect_batch_size(log_plan, batch_size));
  }
  if (OB_SUCC(ret) && batch_size > 0) {
    log_plan.get_optimizer_context().set_batch_size(batch_size);
    phy_plan.set_batch_size(batch_size);
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(generate_exprs(log_plan, phy_plan))) {
  } else if (OB_FAIL(generate_operators(log_plan, phy_plan))) {
  }

  return ret;
}
//1. Generate the old execution plan, used for initializing all expression operators
//   and initialize to ObExpr's op_, for mixed running of new and old expressions, will be removed later
//2. Get all expressions that will be used during execution
//3. Generate all physical expressions
int ObCodeGenerator::generate_exprs(const ObLogPlan &log_plan,
                                    ObPhysicalPlan &phy_plan)
{
  int ret = OB_SUCCESS;
  ObExecContext *exec_ctx = log_plan.get_optimizer_context().get_exec_ctx();
  CK(NULL != exec_ctx && NULL != exec_ctx->get_physical_plan_ctx());
  CK(exec_ctx->get_my_session() != NULL);
  if (OB_SUCC(ret)) {
    ObStaticEngineExprCG expr_cg(
        phy_plan.get_allocator(),
        log_plan.get_optimizer_context().get_session_info(),
        exec_ctx->get_sql_ctx()->schema_guard_,
        exec_ctx->get_physical_plan_ctx()->get_original_param_cnt(),
        param_store_->count());
    // init ctx for operator cg
    expr_cg.set_batch_size(phy_plan.get_batch_size());
    if (OB_FAIL(expr_cg.generate(log_plan.get_optimizer_context().get_all_exprs(),
                                 phy_plan.get_expr_frame_info()))) {
    } else {
      phy_plan.get_next_expr_id() = phy_plan.get_expr_frame_info().need_ctx_cnt_;
    }
  }

  return ret;
}

int ObCodeGenerator::generate_operators(const ObLogPlan &log_plan,
                                        ObPhysicalPlan &phy_plan)
{
  int ret = OB_SUCCESS;
  ObStaticEngineCG static_engin_cg;
  if (OB_FAIL(static_engin_cg.generate(log_plan, phy_plan))) {
  }
  return ret;
}

int ObCodeGenerator::detect_batch_size(
    const ObLogPlan &log_plan, int64_t &batch_size)
{
  int ret = OB_SUCCESS;
  bool vectorize = false;
  bool stop_checking = false;
  batch_size = 0;
  ObBasicSessionInfo *session =
      log_plan.get_optimizer_context().get_session_info();
  ObExecContext *exec_ctx = log_plan.get_optimizer_context().get_exec_ctx();
  CK(NULL != exec_ctx && NULL != exec_ctx->get_physical_plan_ctx());
  bool has_registered_vec_op = false;
  if (OB_ISNULL(log_plan.get_plan_root())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ObStaticEngineCG::exist_registered_vec_op(*log_plan.get_plan_root(), true,
                                                               has_registered_vec_op))) {
  }
  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(session)) {
    // empty session disable batch processing
  } else {
    
    double scan_cardinality = 0;
    // TODO bin.lb: move to optimizer and more sophisticated rules
    int64_t lob_rowsets_max_rows = config::_lob_rowsets_max_rows();
    const ObOptParamHint *opt_params = &log_plan.get_stmt()->get_query_ctx()->get_global_hint().opt_params_;
    if (OB_FAIL(opt_params->get_integer_opt_param(ObOptParamHint::LOB_ROWSETS_MAX_ROWS, lob_rowsets_max_rows))) {
    } else {
      // TODO bin.lb; check all sub plans
      OZ(ObStaticEngineCG::check_vectorize_supported(vectorize,
                                                     stop_checking,
                                                     scan_cardinality,
                                                     log_plan.get_plan_root()));
    }
    if (OB_FAIL(ret)) {
    } else if (log_plan.get_optimizer_context().has_var_assign()
               && !log_plan.get_optimizer_context().is_var_assign_only_in_root_stmt()) {
      // Assignments in a child statement may feed a parent statement without being
      // projected by the child. Keep the whole plan scalar so their row-by-row side
      // effects cannot be reused by batch evaluation.
      batch_size = 0;
    } else if (log_plan.get_optimizer_context().has_var_assign() || !vectorize) {
      // Retain the historical single-row batch path for root-statement assignments
      // when a registered vector operator exists. This preserves evaluation order
      // without forcing that operator onto an unsupported scalar path.
      // set max_batch_size = 1
      // if all physical operator is not registered as vec op, disable vectorization
      if (has_registered_vec_op) {
        batch_size = 1;
      } else {
        batch_size = 0;
      }
    } else if (vectorize) {
      ObArenaAllocator alloc;
      ObRawExprUniqueSet flattened_exprs(true);
      OZ(flattened_exprs.flatten_and_add_raw_exprs(log_plan.get_optimizer_context()
                                                           .get_all_exprs()));
      ObStaticEngineExprCG expr_cg(
          alloc,
          log_plan.get_optimizer_context().get_session_info(),
          exec_ctx->get_sql_ctx()->schema_guard_,
          exec_ctx->get_physical_plan_ctx()->get_original_param_cnt(),
          0);
      int64_t rowsets_max_rows = config::_rowsets_max_rows();
      OZ(expr_cg.detect_batch_size(flattened_exprs, batch_size,
                                   rowsets_max_rows,
                                   config::_rowsets_target_maxsize(),
                                   scan_cardinality,
                                   lob_rowsets_max_rows));
      // Preserve explicit ROWSETS hints. Only replace the small-document
      // scheduling heuristic, without changing logical/cost cardinalities.
      bool has_rowsets_hint = false;
      uint64_t fts_ddl_table_id = common::OB_INVALID_ID;
      if (OB_SUCC(ret) && batch_size == ObStaticEngineExprCG::SMALL_SCAN_CARDINALITY
          && expr_cg.is_small_scan(scan_cardinality)
          && OB_SUCCESS == opt_params->has_opt_param(ObOptParamHint::ROWSETS_MAX_ROWS, has_rowsets_hint)
          && !has_rowsets_hint && can_expand_fts_term_batch(log_plan, fts_ddl_table_id)) {
        int64_t normal_cap = batch_size;
        const double scheduling_cardinality = ObStaticEngineExprCG::SMALL_SCAN_CARDINALITY;
        const int probe_ret = expr_cg.detect_batch_size(flattened_exprs, normal_cap,
            rowsets_max_rows, config::_rowsets_target_maxsize(), scheduling_cardinality,
            lob_rowsets_max_rows);
        const int64_t frame_budget = common::min(512 * 1024, config::_rowsets_target_maxsize());
        int64_t frame_bound = 0;
        const int64_t original_batch = batch_size;
        const int64_t parameter_count = exec_ctx->get_physical_plan_ctx()->get_datum_param_store().count();
        const int64_t original_parameter_count = exec_ctx->get_physical_plan_ctx()->get_original_param_cnt();
        if (OB_SUCCESS == probe_ret && normal_cap >= 256
            && fts_expr_frames_fit(flattened_exprs, 256, frame_budget, lob_rowsets_max_rows,
                                   parameter_count, original_parameter_count, frame_bound)) {
          batch_size = 256;
        } else if (OB_SUCCESS == probe_ret && normal_cap >= 128
            && fts_expr_frames_fit(flattened_exprs, 128, frame_budget, lob_rowsets_max_rows,
                                   parameter_count, original_parameter_count, frame_bound)) {
          batch_size = 128;
        }
        LOG_INFO("FTS term expansion frame preflight", K(original_batch), K(batch_size),
                 K(normal_cap), K(frame_budget), K(frame_bound), K(scan_cardinality), K(fts_ddl_table_id));
      }
      // overwrite batch size if hint is specified
      OZ(opt_params->get_integer_opt_param(ObOptParamHint::ROWSETS_MAX_ROWS, batch_size));

      int tmp_ret = OB_SUCCESS;
      tmp_ret = OB_E(EventTable::EN_ENABLE_RANDOM_BATCH_SIZE) OB_SUCCESS;
      if (OB_SUCCESS != tmp_ret) {
        static const int64_t min = 1;
        static const int64_t max = rowsets_max_rows * 2;
        batch_size = common::ObRandom::rand(min, max);
      }
    }
    // TODO qubin.qb: remove the tracelog when rowsets/batch_size is displayed
    // in plan
  }
  return ret;
}

} // end namespace sql
} // end namespace oceanbase
