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

#define USING_LOG_PREFIX SQL_ENG

#include "sql/engine/sort/ob_sort_op.h"
#include "sql/engine/px/ob_px_util.h"
#include "sql/engine/aggregate/ob_hash_groupby_op.h"
#include "sql/engine/expr/ob_expr_topn_filter.h"
#include "sql/ob_sql_context.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"
#include "sql/engine/px/ob_px_sqc_handler.h"
#include "sql/engine/px/exchange/ob_px_receive_op.h"
#include "sql/dtl/ob_dtl_basic_channel.h"

namespace oceanbase
{
namespace sql
{

ObSortSpec::ObSortSpec(common::ObIAllocator &alloc, const ObPhyOperatorType type)
  : ObOpSpec(alloc, type),
  topn_expr_(nullptr),
  topk_limit_expr_(nullptr),
  topk_offset_expr_(nullptr),
  all_exprs_(alloc),
  sort_collations_(alloc),
  sort_cmp_funs_(alloc),
  minimum_row_count_(0),
  topk_precision_(0),
  prefix_pos_(0),
  is_local_merge_sort_(false),
  is_fetch_with_ties_(false),
  prescan_enabled_(false),
  fts_single_pass_plan_(false),
  enable_encode_sortkey_opt_(false),
  part_cnt_(0),
  pd_topn_filter_info_(alloc)
{}

OB_SERIALIZE_MEMBER((ObSortSpec, ObOpSpec),
                    topn_expr_,
                    topk_limit_expr_,
                    topk_offset_expr_,
                    all_exprs_,
                    sort_collations_,
                    sort_cmp_funs_,
                    minimum_row_count_,
                    topk_precision_,
                    prefix_pos_,
                    is_local_merge_sort_,
                    is_fetch_with_ties_,
                    prescan_enabled_,
                    enable_encode_sortkey_opt_,
                    part_cnt_,
                    compress_type_,
                    pd_topn_filter_info_,
                    fts_single_pass_plan_);

ObSortOp::ObSortOp(ObExecContext &ctx_, const ObOpSpec &spec, ObOpInput *input)
  : ObOperator(ctx_, spec, input),
  sort_impl_(op_monitor_info_),
  prefix_sort_impl_(op_monitor_info_),
  read_func_(&ObSortOp::sort_impl_next),
  read_batch_func_(&ObSortOp::sort_impl_next_batch),
  sort_row_count_(0),
  is_first_(true),
  ret_row_count_(0),
  iter_end_(false)
{}

int ObSortOp::inner_open()
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(child_)) {
    ret = OB_ERR_UNEXPECTED;
  }
  return OB_SUCCESS;
}

int ObSortOp::inner_rescan()
{
  if (MY_SPEC.enable_pd_topn_filter() && !MY_SPEC.pd_topn_filter_info_.is_shuffle_) {
    // for local topn runtime filter, rescan topn filter expression context
    reset_pd_topn_filter_expr_ctx();
  }
  reset();
  iter_end_ = false;
  return ObOperator::inner_rescan();
}

void ObSortOp::reset_pd_topn_filter_expr_ctx()
{
  uint32_t expr_ctx_id = MY_SPEC.pd_topn_filter_info_.expr_ctx_id_;
  ObExprTopNFilterContext *topn_filter_ctx =
      static_cast<ObExprTopNFilterContext *>(ctx_.get_expr_op_ctx(expr_ctx_id));
  if (nullptr != topn_filter_ctx) {
    topn_filter_ctx->reset_for_rescan();
  }
}

void ObSortOp::reset()
{
  sort_impl_.reset();
  prefix_sort_impl_.reset();
  read_func_ = &ObSortOp::sort_impl_next;
  read_batch_func_ = &ObSortOp::sort_impl_next_batch;
  sort_row_count_ = 0;
  ret_row_count_ = 0;
  is_first_ = true;
}

void ObSortOp::destroy()
{
  sort_impl_.unregister_profile_if_necessary();
  sort_impl_.~ObSortOpImpl();
  prefix_sort_impl_.unregister_profile_if_necessary();
  prefix_sort_impl_.~ObPrefixSortImpl();
  read_func_ = nullptr;
  read_batch_func_ = nullptr;
  sort_row_count_ = 0;
  is_first_ = true;
  ret_row_count_ = 0;
  ObOperator::destroy();
}

int ObSortOp::inner_close()
{
  sort_impl_.collect_memory_dump_info(op_monitor_info_);
  sort_impl_.unregister_profile();
  prefix_sort_impl_.unregister_profile();
  return OB_SUCCESS;
}

int ObSortOp::get_int_value(const ObExpr *in_val, int64_t &out_val)
{
  int ret = OB_SUCCESS;
  ObDatum *datum = NULL;
  if (NULL != in_val) {
    if (OB_FAIL(in_val->eval(eval_ctx_, datum))) {
    } else if (OB_ISNULL(datum)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (datum->is_null()) {
      out_val = 0;
    } else {
      out_val = *datum->int_;
    }
  }
  return ret;
}

int ObSortOp::get_topn_count(int64_t &topn_cnt)
{
  int ret = OB_SUCCESS;
  topn_cnt = INT64_MAX;
  if ((OB_ISNULL(MY_SPEC.topn_expr_) && OB_ISNULL(MY_SPEC.topk_limit_expr_))) {
    // do nothing
  } else if (((NULL != MY_SPEC.topn_expr_) && (NULL != MY_SPEC.topk_limit_expr_))) {
    ret = OB_ERR_UNEXPECTED;
  } else if (NULL != MY_SPEC.topn_expr_) {
    if (OB_FAIL(get_int_value(MY_SPEC.topn_expr_, topn_cnt))) {
    } else {
      topn_cnt = std::max(MY_SPEC.minimum_row_count_, topn_cnt);
    }
  } else if (NULL != MY_SPEC.topk_limit_expr_) {
    int64_t limit = -1;
    int64_t offset = 0;
    if ((OB_FAIL(get_int_value(MY_SPEC.topk_limit_expr_, limit))
        || OB_FAIL(get_int_value(MY_SPEC.topk_offset_expr_, offset)))) {
    } else if (OB_UNLIKELY(limit < 0 || offset < 0)) {
      ret = OB_ERR_ILLEGAL_VALUE;
    } else {
      // TODO & FIXME by longzhong.wlz : Wait for groupby implementation to handle this logic
      topn_cnt = std::max(MY_SPEC.minimum_row_count_, limit + offset);
      int64_t row_count = 0;
      ObPhyOperatorType op_type = child_->get_spec().type_;
      if (PHY_HASH_GROUP_BY != op_type) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        row_count = static_cast<ObHashGroupByOp *>(child_)->get_hash_groupby_row_count();
      }
      if (OB_SUCC(ret)) {
        topn_cnt = std::max(topn_cnt,
                            static_cast<int64_t>(row_count * MY_SPEC.topk_precision_ / 100));
        if (topn_cnt >= row_count) {
          ctx_.get_physical_plan_ctx()->set_is_result_accurate(true);
        } else {
          ctx_.get_physical_plan_ctx()->set_is_result_accurate(false);
        }
      }
    }
  }
  return ret;
}

namespace
{
bool can_single_pass_fts_sort(const ObSortSpec &spec, ObExecContext &ctx,
    ObOperator *child, ObOperator *parent, const char *&reason, int64_t &source_rows);
}

int ObSortOp::process_sort()
{
  int ret = OB_SUCCESS;
  if (read_func_ == &ObSortOp::prefix_sort_impl_next) {
    // prefix sort get child row in it's own wrap, do nothing here
  } else if (read_func_ == &ObSortOp::sort_impl_next) {
    bool need_dump = false;
    bool single_pass = false, single_pass_checked = false;
    int64_t source_rows = -1;
    while (OB_SUCC(ret)) {
      clear_evaluated_flag();
      if (OB_FAIL(try_check_status())) {
      } else if (OB_FAIL(child_->get_next_row())) {
        if (OB_ITER_END != ret) {
        }
      } else {
        sort_row_count_++;
        OZ(sort_impl_.add_row(MY_SPEC.all_exprs_, need_dump));
        sort_impl_.collect_memory_dump_info(op_monitor_info_);
        if (OB_SUCC(ret) && need_dump && MY_SPEC.prescan_enabled_) {
          if (!single_pass_checked) {
            single_pass_checked = true;
            const char *reason = nullptr;
            single_pass = can_single_pass_fts_sort(MY_SPEC, ctx_, child_, get_parent(), reason, source_rows);
            LOG_INFO("FTS single-pass sort decision", K(single_pass), K(reason),
                "certificate", MY_SPEC.fts_single_pass_plan_, K(sort_row_count_), K(source_rows));
          }
          if (!single_pass) break;
        }
      }
    }
    if (OB_SUCC(ret) && need_dump && MY_SPEC.prescan_enabled_ && !single_pass
        && OB_FAIL(scan_all_then_sort())) {
      if (OB_ITER_END != ret) {
      }
    }
    if (OB_ITER_END == ret) {
      ret = OB_SUCCESS;
    }
    OZ(sort_impl_.sort());
    sort_impl_.collect_memory_dump_info(op_monitor_info_);
    if (single_pass) LOG_INFO("FTS single-pass sort complete", K(ret), K(sort_row_count_), K(source_rows));
  } else {
    ret = OB_ERR_UNEXPECTED;
  }
  return ret;
}

int ObSortOp::process_sort_batch()
{
  int ret = OB_SUCCESS;
  if (read_batch_func_ == &ObSortOp::prefix_sort_impl_next_batch) {
    // prefix sort get child row in it's own wrap, do nothing here
  } else if (read_batch_func_ == &ObSortOp::sort_impl_next_batch) {
    bool need_dump = false;
    bool single_pass = false, single_pass_checked = false;
    int64_t source_rows = -1;
    while (OB_SUCC(ret)) {
      clear_evaluated_flag();
      const ObBatchRows *input_brs = NULL;
      if (OB_FAIL(try_check_status())) {
      } else if (OB_FAIL(child_->get_next_batch(MY_SPEC.max_batch_size_, input_brs))) {
      } else {
        if (input_brs->size_ > 0) {
          sort_row_count_ += input_brs->size_
              - input_brs->skip_->accumulate_bit_cnt(input_brs->size_);
          OZ(sort_impl_.add_batch(MY_SPEC.all_exprs_, *input_brs->skip_,
                                  input_brs->size_, need_dump));
          sort_impl_.collect_memory_dump_info(op_monitor_info_);
        }
        if (input_brs->end_) break;
        if (OB_SUCC(ret) && need_dump && MY_SPEC.prescan_enabled_) {
          if (!single_pass_checked) {
            single_pass_checked = true;
            const char *reason = nullptr;
            single_pass = can_single_pass_fts_sort(MY_SPEC, ctx_, child_, get_parent(), reason, source_rows);
            LOG_INFO("FTS single-pass sort decision", K(single_pass), K(reason),
                "certificate", MY_SPEC.fts_single_pass_plan_, K(sort_row_count_), K(source_rows));
          }
          if (!single_pass) break;
        }
      }
    }
    if (OB_SUCC(ret) && need_dump && MY_SPEC.prescan_enabled_ && !single_pass
        && OB_FAIL(scan_all_then_sort_batch())) {
      if (OB_ITER_END != ret) {
      }
    }
    op_monitor_info_.otherstat_7_id_ = ObSqlMonitorStatIds::ROW_COUNT;
    op_monitor_info_.otherstat_7_value_ = sort_row_count_; 
    OZ(sort_impl_.sort());
    sort_impl_.collect_memory_dump_info(op_monitor_info_);
    if (single_pass) LOG_INFO("FTS single-pass sort complete", K(ret), K(sort_row_count_), K(source_rows));
  } else {
    ret = OB_ERR_UNEXPECTED;
  }
  return ret;
}

namespace
{
bool can_replay_fts_cache(const ObSortSpec &spec, ObExecContext &ctx,
    int64_t &doc_col, const char *&decline_reason, int64_t &decline_expr,
    int &formal_guard_ret, bool &is_fts_target)
{
  doc_col = -1;
  is_fts_target = false;
  decline_expr = -1;
  decline_reason = "sort mode";
  if (!spec.prescan_enabled_ || spec.enable_encode_sortkey_opt_
      || spec.prefix_pos_ != 0 || spec.part_cnt_ != 0 || spec.is_local_merge_sort_
      || spec.is_fetch_with_ties_ || nullptr != spec.topn_expr_ || nullptr != spec.topk_limit_expr_) return false;
  decline_reason = "DDL context";
  if (nullptr == spec.plan_ || spec.plan_->get_ddl_task_id() <= 0
      || nullptr == ctx.get_my_session() || !ctx.get_my_session()->get_ddl_info().is_ddl()) return false;
  ObSqlCtx *sql_ctx = ctx.get_sql_ctx();
  const share::schema::ObTableSchema *target = nullptr;
  decline_reason = "schema guard";
  if (nullptr == sql_ctx || nullptr == sql_ctx->schema_guard_) return false;
  // This optimization copies opaque datums; the runtime guard pins the actual
  // target schema. Global version-number formalness is diagnostic, not layout.
  formal_guard_ret = sql_ctx->schema_guard_->check_formal_guard();
  decline_reason = "target lookup";
  if (common::OB_SUCCESS != sql_ctx->schema_guard_->get_table_schema(spec.plan_->get_ddl_table_id(), target)) return false;
  decline_reason = "target identity";
  if (nullptr == target || !target->is_valid()
      || target->get_table_id() != spec.plan_->get_ddl_table_id()) return false;
  decline_reason = "target is not FTS";
  if (!target->is_fts_index_aux() && !target->is_fts_doc_word_aux()) return false;
  is_fts_target = true;
  decline_reason = "column count";
  if (spec.all_exprs_.count() != 4 && spec.all_exprs_.count() != 5) return false;
  int64_t strings = 0, ids = 0, integers = 0;
  for (int64_t i = 0; i < spec.all_exprs_.count(); ++i) {
    const ObExpr *e = spec.all_exprs_.at(i);
    decline_expr = i;
    decline_reason = "null or LOB expression";
    if (nullptr == e || e->obj_meta_.has_lob_header()) return false;
    decline_reason = "duplicate expression";
    for (int64_t j = 0; j < i; ++j) if (e == spec.all_exprs_.at(j)) return false;
    decline_reason = "expression type/map";
    if (e->datum_meta_.type_ == common::ObVarcharType
        && e->obj_datum_map_ == common::OBJ_DATUM_STRING) {
      ++strings;
      if (e->datum_meta_.cs_type_ == common::CS_TYPE_BINARY) {
        ++ids;
        doc_col = i;
      }
    } else if ((e->datum_meta_.type_ == common::ObIntType
                || e->datum_meta_.type_ == common::ObUInt64Type)
        && e->obj_datum_map_ == common::OBJ_DATUM_8BYTE_DATA) {
      ++integers;
    } else return false;
  }
  decline_expr = -1;
  decline_reason = "column layout counts";
  // Two binary strings are ambiguous: keep the original path for that layout.
  if (strings != 2 || ids != 1 || integers != spec.all_exprs_.count() - 2) return false;
  decline_reason = "enabled";
  return true;
}

bool can_single_pass_fts_sort(const ObSortSpec &spec, ObExecContext &ctx,
    ObOperator *child, ObOperator *parent, const char *&reason, int64_t &source_rows)
{
  reason = "full-tree certificate";
  if (!spec.fts_single_pass_plan_ || nullptr == spec.plan_
      || spec.plan_->get_px_dop() != 1 || spec.plan_->get_is_use_auto_dop()
      || ctx.is_use_adaptive_px_dop()) return false;
  int64_t doc_col = -1, declined_expr = -1;
  int formal_ret = common::OB_SUCCESS;
  bool is_fts = false;
  // Keep the existing exact target identity / opaque non-LOB tuple gate.
  if (!can_replay_fts_cache(spec, ctx, doc_col, reason, declined_expr, formal_ret, is_fts)) return false;
  reason = "runtime ordered writer / FIFO";
  if (nullptr == parent || parent->get_spec().type_ != PHY_PX_MULTI_PART_SSTABLE_INSERT
      || parent->get_child_cnt() != 1 || parent->get_child() == nullptr
      || &parent->get_child()->get_spec() != &spec || nullptr == child
      || child->get_spec().type_ != PHY_PX_FIFO_RECEIVE || child->get_child_cnt() != 0) return false;
  ObPxSqcHandler *handler = ctx.get_sqc_handler();
  reason = "single streaming SQC actual / maximum degree";
  if (nullptr == handler) return false;
  ObPxSqcMeta &sqc = handler->get_sqc_init_arg().sqc_;
  if (sqc.is_fulltree() || sqc.get_task_count() != 1 || sqc.get_max_task_count() != 1
      || sqc.get_min_task_count() != 1 || sqc.get_total_task_count() != 1
      || sqc.recieve_use_interm_result() || sqc.transmit_use_interm_result()
      || sqc.get_serial_receive_channels().count() != 0) {
    LOG_INFO("FTS single-pass SQC declined", "fulltree", sqc.is_fulltree(),
        "tasks", sqc.get_task_count(), "max_tasks", sqc.get_max_task_count(),
        "min_tasks", sqc.get_min_task_count(), "total_tasks", sqc.get_total_task_count(),
        "receive_interm", sqc.recieve_use_interm_result(), "transmit_interm", sqc.transmit_use_interm_result(),
        "serial_channels", sqc.get_serial_receive_channels().count());
    return false;
  }
  auto *input = static_cast<ObPxReceiveOpInput *>(child->get_input());
  reason = "one linked streaming FIFO channel";
  auto *receive = static_cast<ObPxFifoReceiveOp *>(child);
  if (nullptr == input || !receive->channel_linked() || receive->get_ch_set().count() != 1
      || receive->get_task_channels().count() != 1) {
    LOG_INFO("FTS single-pass streaming channel declined", "has_input", nullptr != input,
        "linked", receive->channel_linked(), "channel_set", receive->get_ch_set().count(),
        "task_channels", receive->get_task_channels().count());
    return false;
  }
  dtl::ObDtlChannel *channel = receive->get_task_channels().at(0);
  if (nullptr == channel || channel->use_interm_result()) return false;
  // The complete-tree certificate requires RANGE/OBJECT_SAMPLE. That transmit
  // obtains the sample hub result, then rescans its STAT_COLLECTOR (resetting
  // its internal sample sort and enabling bypass) before send_rows. Reaching
  // this already-owned prefix therefore proves that barrier has finished.
  // The remaining source is SCAN -> SUBPLAN -> bypass STAT -> one stream.
  // Our quota-controlled spill owns its prefix and fd and never waits on that
  // producer; after each spill we keep draining its credits until real EOF.
  source_rows = -1; // Streaming input has no sealed store row-count oracle.
  reason = "single stream after range-sampling barrier";
  return true;
}

int replay_fts_sort_cache(ObCompactStore &cache_store, const ObSortSpec &spec,
    ObExecContext &ctx, ObSortOpImpl &sort_impl, const char *entry)
{
  int ret = OB_SUCCESS;
  int64_t doc_col = -1, decline_expr = -1, view_rows = 0, fallback_rows = 0;
  int64_t layout_fallback_rows = 0, doc_id_fallback_rows = 0;
  int formal_guard_ret = OB_NOT_INIT;
  bool is_fts_target = false;
  const char *decline_reason = nullptr;
  const bool eligible = can_replay_fts_cache(spec, ctx, doc_col, decline_reason, decline_expr,
                                           formal_guard_ret, is_fts_target);
  const ObChunkDatumStore::StoredRow *store_row = nullptr;
  bool has_next = false;
  while (OB_SUCC(ret) && OB_SUCC(cache_store.has_next(has_next)) && has_next) {
    common::ObDatum views[5];
    int64_t compact_size = 0;
    bool supported = false;
    if (eligible && OB_FAIL(cache_store.peek_fts_datums(
        views, spec.all_exprs_.count(), compact_size, supported))) {
      break;
    } else if (supported && views[doc_col].len_ == common::OB_DOC_ID_COLUMN_BYTE_LENGTH) {
      if (OB_FAIL(sort_impl.add_fts_datums(views, spec.all_exprs_.count()))) {
        break; // no retry or source advance after a destination failure
      }
      cache_store.consume_fts_row(compact_size);
      ++view_rows;
      continue;
    }
    if (eligible) {
      if (supported) ++doc_id_fallback_rows;
      else ++layout_fallback_rows;
    }
    ++fallback_rows;
    if (OB_FAIL(cache_store.get_next_row(store_row))) {
      if (OB_ITER_END != ret) {
        LOG_WARN("failed to get next row", K(ret));
      }
    } else if (OB_ISNULL(store_row)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      OZ(sort_impl.add_stored_row(*store_row));
    }
  }
  const int64_t decline_expr_type = decline_expr < 0 || nullptr == spec.all_exprs_.at(decline_expr)
      ? -1 : static_cast<int64_t>(spec.all_exprs_.at(decline_expr)->type_);
  LOG_INFO("FTS prescan datum replay", K(ret), K(entry), K(eligible), K(decline_reason),
      K(decline_expr), K(decline_expr_type), K(formal_guard_ret), K(is_fts_target),
      "max_batch_size", spec.max_batch_size_,
      K(view_rows), K(fallback_rows), K(layout_fallback_rows), K(doc_id_fallback_rows),
      "cached_rows", cache_store.get_row_cnt());
  if (!eligible && is_fts_target) {
    // One metadata record per expression at this prescan's first rejection;
    // never log text/token values or run this diagnostic in the row loop.
    for (int64_t i = 0; i < spec.all_exprs_.count(); ++i) {
      const ObExpr *e = spec.all_exprs_.at(i);
      LOG_INFO("FTS prescan replay declined expression", K(entry), K(i),
          "type", nullptr == e ? -1 : static_cast<int64_t>(e->type_),
          "datum_type", nullptr == e ? -1 : static_cast<int64_t>(e->datum_meta_.type_),
          "collation", nullptr == e ? -1 : static_cast<int64_t>(e->datum_meta_.cs_type_),
          "datum_map", nullptr == e ? -1 : static_cast<int64_t>(e->obj_datum_map_),
          "has_lob_header", nullptr != e && e->obj_meta_.has_lob_header());
    }
  }
  return ret;
}
} // namespace

// if we need to do std sort, the thread will be blocked and cannot
// drive the table scan below, which will block other sort ops.
// To relieve this, we scan all rows into a cache store first then
// continue the sort part.
int ObSortOp::scan_all_then_sort()
{
  int ret = OB_SUCCESS;
  SMART_VAR(ObCompactStore, cache_store) {
    if (OB_FAIL(cache_store.init(2 * 1024 * 1024,
        ObCtxIds::DEFAULT_CTX_ID, "SORT_CACHE_CTX", true/*enable dump*/, 0, true,
        MY_SPEC.compress_type_, &MY_SPEC.all_exprs_))) {
    } else if (OB_FAIL(cache_store.alloc_dir_id())) {
    }
    while (OB_SUCC(ret)) {
      clear_evaluated_flag();
      if (OB_FAIL(try_check_status())) {
      } else if (OB_FAIL(child_->get_next_row())) {
        if (OB_ITER_END != ret) {
        }
      } else {
        sort_row_count_++;
        if (OB_FAIL(cache_store.add_row(MY_SPEC.all_exprs_, eval_ctx_))) {
        }
      }
    }

    if (OB_ITER_END == ret) {
      ret = OB_SUCCESS;
    }

    if (OB_SUCC(ret)) {
      if (OB_FAIL(cache_store.finish_add_row(false))) {
      } else {
        OZ(replay_fts_sort_cache(cache_store, MY_SPEC, ctx_, sort_impl_, "scalar"));
      }
    }
  }
  return ret;
}

int ObSortOp::scan_all_then_sort_batch()
{
  int ret = OB_SUCCESS;
  SMART_VAR(ObCompactStore, cache_store) {
    if (OB_FAIL(cache_store.init(16 * 1024,
        ObCtxIds::DEFAULT_CTX_ID, "SORT_CACHE_CTX", true/*enable dump*/, 0, true,
        MY_SPEC.compress_type_, &MY_SPEC.all_exprs_))) {
    } else if (OB_FAIL(cache_store.alloc_dir_id())) {
    }
    while (OB_SUCC(ret)) {
      clear_evaluated_flag();
      const ObBatchRows *input_brs = NULL;
      if (OB_FAIL(try_check_status())) {
      } else if (OB_FAIL(child_->get_next_batch(MY_SPEC.max_batch_size_, input_brs))) {
      } else {
        if (input_brs->size_ > 0) {
          sort_row_count_ += input_brs->size_
              - input_brs->skip_->accumulate_bit_cnt(input_brs->size_);
          int64_t stored_row_count = -1;
          if (OB_FAIL(cache_store.add_batch(MY_SPEC.all_exprs_, eval_ctx_,
              *input_brs->skip_, input_brs->size_, stored_row_count))) {
          }
        }
        if (input_brs->end_) {
          break;
        }
      }
    }

    if (OB_ITER_END == ret) {
      ret = OB_SUCCESS;
    }
    op_monitor_info_.otherstat_7_id_ = ObSqlMonitorStatIds::ROW_COUNT;
    op_monitor_info_.otherstat_7_value_ = sort_row_count_; 
    if (OB_SUCC(ret)) {
      if (OB_FAIL(cache_store.finish_add_row(false))) {
      } else {
        OZ(replay_fts_sort_cache(cache_store, MY_SPEC, ctx_, sort_impl_, "batch"));
      }
    }
  }
  return ret;
}

int ObSortOp::init_prefix_sort(int64_t row_count,
                               bool is_batch,
                               int64_t topn_cnt)
{
  int ret = OB_SUCCESS;
  OZ(prefix_sort_impl_.init(MY_SPEC.prefix_pos_, MY_SPEC.all_exprs_,
      &MY_SPEC.sort_collations_, &MY_SPEC.sort_cmp_funs_, &eval_ctx_, child_,
      this, ctx_, MY_SPEC.enable_encode_sortkey_opt_, sort_row_count_, topn_cnt,
      MY_SPEC.is_fetch_with_ties_));
  if (is_batch) {
    read_batch_func_ = &ObSortOp::prefix_sort_impl_next_batch;
  } else {
    read_func_ = &ObSortOp::prefix_sort_impl_next;
  }
  int aqs_head = MY_SPEC.enable_encode_sortkey_opt_ ? sizeof(oceanbase::sql::ObSortOpImpl::AQSItem) : 0;
  prefix_sort_impl_.set_input_rows(row_count);
  prefix_sort_impl_.set_input_width(MY_SPEC.width_ + aqs_head);
  prefix_sort_impl_.set_operator_type(MY_SPEC.type_);
  prefix_sort_impl_.set_operator_id(MY_SPEC.id_);
  prefix_sort_impl_.set_io_event_observer(&io_event_observer_);
  return ret;
}

int ObSortOp::init_sort(int64_t row_count,
                        bool is_batch,
                        int64_t topn_cnt)
{
  int ret = OB_SUCCESS;
  int64_t est_rows = MY_SPEC.rows_;
  if (OB_FAIL(ObPxEstimateSizeUtil::get_px_size(
      &ctx_, MY_SPEC.px_est_size_factor_, est_rows, est_rows))) {
  }
  OZ(sort_impl_.init(&MY_SPEC.sort_collations_, &MY_SPEC.sort_cmp_funs_, &eval_ctx_,
                     &ctx_, MY_SPEC.enable_encode_sortkey_opt_, MY_SPEC.is_local_merge_sort_,
                     false /* need_rewind */, MY_SPEC.part_cnt_, topn_cnt,
                     MY_SPEC.is_fetch_with_ties_, ObChunkDatumStore::BLOCK_SIZE,
                     MY_SPEC.compress_type_, &MY_SPEC.all_exprs_, est_rows,
                     MY_SPEC.prescan_enabled_, &MY_SPEC.pd_topn_filter_info_));
  // Resolve the real DDL target on the executing PX worker. A tuple's type
  // pattern alone never opts a sort in; forward/doc-word auxiliaries fall back.
  if (OB_SUCC(ret) && MY_SPEC.prescan_enabled_ && !MY_SPEC.enable_encode_sortkey_opt_
      && 0 == MY_SPEC.prefix_pos_ && 0 == MY_SPEC.part_cnt_
      && nullptr == MY_SPEC.topn_expr_ && nullptr == MY_SPEC.topk_limit_expr_
      && !MY_SPEC.is_local_merge_sort_ && !MY_SPEC.is_fetch_with_ties_
      && nullptr != MY_SPEC.plan_ && MY_SPEC.plan_->get_ddl_task_id() > 0
      && ctx_.get_my_session()->get_ddl_info().is_ddl()) {
    ObSqlCtx *sql_ctx = ctx_.get_sql_ctx();
    const share::schema::ObTableSchema *target = nullptr;
    int schema_ret = OB_SUCCESS;
    bool enabled = false;
    if (nullptr != sql_ctx && nullptr != sql_ctx->schema_guard_
        && OB_SUCCESS == (schema_ret = sql_ctx->schema_guard_->get_table_schema(
            MY_SPEC.plan_->get_ddl_table_id(), target)) && nullptr != target
        && target->is_fts_index_aux()) {
      enabled = sort_impl_.enable_fts_term_rank_sort();
    }
    LOG_INFO("FTS run-local term rank eligibility", K(enabled), K(schema_ret),
             "ddl_table_id", MY_SPEC.plan_->get_ddl_table_id(),
             "sort_key_count", MY_SPEC.sort_collations_.count());
  }
  if (is_batch) {
    read_batch_func_ = &ObSortOp::sort_impl_next_batch;
  } else {
    read_func_ = &ObSortOp::sort_impl_next;
  }
  int aqs_head = MY_SPEC.enable_encode_sortkey_opt_ ? sizeof(oceanbase::sql::ObSortOpImpl::AQSItem) : 0;
  sort_impl_.set_input_rows(row_count);
  sort_impl_.set_input_width(MY_SPEC.width_ + aqs_head);
  sort_impl_.set_operator_type(MY_SPEC.type_);
  sort_impl_.set_operator_id(MY_SPEC.id_);
  sort_impl_.set_io_event_observer(&io_event_observer_);
  return ret;
}

int ObSortOp::inner_get_next_row()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(iter_end_)) {
    ret = OB_ITER_END;
  } else if (is_first_) {
    // Here what we want is to account
    // Charge memory to the runtime that owns the execution context.
    is_first_ = false;
    int64_t topn_cnt = INT64_MAX;
    int64_t row_count = MY_SPEC.rows_;
    
    if (OB_FAIL(ObPxEstimateSizeUtil::get_px_size(
        &ctx_, MY_SPEC.px_est_size_factor_, MY_SPEC.rows_, row_count))) {
    } else if (OB_FAIL(get_topn_count(topn_cnt))) {
    } else if (topn_cnt <= 0) { 
      iter_end_ = true; 
      ret = OB_ITER_END;
    } else if (MY_SPEC.prefix_pos_ > 0) {
      if (OB_FAIL(init_prefix_sort(row_count, false, topn_cnt))) {
      }
    } else {
      if (OB_FAIL(init_sort(row_count, false, topn_cnt))) {
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(process_sort())) { // process sort
        if (OB_ITER_END != ret) {
        }
      }
    }
  }

  if (OB_SUCC(ret)) {
    clear_evaluated_flag();
    if (OB_FAIL((this->*read_func_)())) {
      if (OB_ITER_END != ret) {
        LOG_WARN("get next row failed");
      } else {
        if (ctx_.get_my_session()->get_ddl_info().is_ddl() && ret_row_count_ != sort_row_count_) {
          ret = OB_CHECKSUM_ERROR;
        }
        iter_end_ = true;
        reset();
      }
    } else {
      ++ret_row_count_;
      LOG_DEBUG("finish ObSortOp::inner_get_next_row", K(ObToStringExprRow(eval_ctx_, MY_SPEC.output_)), K(ret_row_count_), K(MY_SPEC.output_));
    }
  }
  return ret;
}

int ObSortOp::inner_get_next_batch(const int64_t max_row_cnt)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(iter_end_)) {
    brs_.end_ = true;
    brs_.size_ = 0;
  } else if (is_first_) {
    is_first_ = false;
    int64_t topn_cnt = INT64_MAX;
    int64_t row_count = MY_SPEC.rows_;
    
    if (OB_FAIL(ObPxEstimateSizeUtil::get_px_size(
        &ctx_, MY_SPEC.px_est_size_factor_, MY_SPEC.rows_, row_count))) {
    } else if (OB_FAIL(get_topn_count(topn_cnt))) {
    } else if (topn_cnt <= 0) { 
      brs_.end_ = true;
      brs_.size_ = 0;
    } else if (MY_SPEC.prefix_pos_ > 0) {
      if (OB_FAIL(init_prefix_sort(row_count, true, topn_cnt))) {
      }
    } else {
      if (OB_FAIL(init_sort(row_count, true, topn_cnt))) {
      }
    }
    if (OB_SUCC(ret) && !brs_.end_) {
      if (OB_FAIL(process_sort_batch())) {
      }
    }
  }

  if (OB_SUCC(ret) && !brs_.end_) {
    clear_evaluated_flag();
    if (OB_FAIL((this->*read_batch_func_)(std::min(max_row_cnt, MY_SPEC.max_batch_size_)))) {
    } else {
      ret_row_count_ += brs_.size_;
      if (brs_.end_) {
        if (ctx_.get_my_session()->get_ddl_info().is_ddl() && ret_row_count_ != sort_row_count_) {
          ret = OB_CHECKSUM_ERROR;
        }
      }
    }
  }
  return ret;
}

} // end namespace sql
} // end namespace oceanbase
