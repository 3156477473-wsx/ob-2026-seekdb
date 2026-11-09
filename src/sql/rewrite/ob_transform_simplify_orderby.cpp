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

#define USING_LOG_PREFIX SQL_REWRITE
#include "sql/rewrite/ob_transform_simplify_orderby.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
using namespace oceanbase::sql;

/*@brief, remove_stmt_order_by removes order by from stmt that can be removed, eg:
* select * from t1 where c1 in (select c1 from t2 order by c2);
* ==>
* select * from t1 where c1 in (select c1 from t2);
*
* select distinct c1 from (select c1 from t1 order by c2);
* ==>
* select distinct c1 from (select c1 from t1);
*
* select  * from (select * from t1 order by c1) order by c2;
* ==>
* select  * from (select * from t1) order by c2;
*
* select c1 from (select c1 from t1 order by c2) group by c1;
* ==>
* select c1 from (select c1 from t1) group by c1;
*
* select c1 from t1 where c2 in (select c2 from t2 order by c1);
* ==>
* select c1 from t1 where c2 in (select c2 from t2);
*/
bool ObTransformSimplifyOrderby::is_count_over_limited_fts_view(const ObDMLStmt &stmt)
{
  bool valid = false;
  if (stmt.is_select_stmt()) {
    const ObSelectStmt &outer = static_cast<const ObSelectStmt &>(stmt);
    auto is_simple_block = [](const ObSelectStmt &s) {
      return s.is_single_table_stmt() && !s.get_from_item(0).is_joined_
          && !s.is_set_stmt() && !s.has_distinct() && !s.has_having()
          && !s.has_window_function() && !s.has_window_function_filter()
          && !s.has_for_update() && !s.has_select_into() && !s.is_calc_found_rows()
          && !s.is_contains_assignment() && s.get_subquery_expr_size() == 0
          && s.get_semi_info_size() == 0 && s.get_cte_definition_size() == 0
          && s.get_pseudo_column_like_exprs().empty()
          && s.get_stmt_hint().get_no_rewrite_hint() == nullptr;
    };
    const TableItem *view = outer.get_table_size() == 1 ? outer.get_table_item(0) : nullptr;
    if (is_simple_block(outer) && !outer.has_limit() && !outer.has_order_by()
        && outer.get_condition_exprs().empty() && outer.get_match_exprs().empty()
        && outer.is_scala_group_by() && outer.get_aggr_item_size() == 1
        && outer.get_select_item_size() == 1 && view != nullptr
        && view->is_generated_table() && !view->is_view_table_
        && view->cte_type_ == TableItem::NOT_CTE && view->exec_params_.empty()
        && view->ref_query_ != nullptr
        && outer.get_from_item(0).table_id_ == view->table_id_) {
      const ObAggFunRawExpr *count = outer.get_aggr_item(0);
      const ObSelectStmt &inner = *view->ref_query_;
      const TableItem *base = inner.get_table_size() == 1 ? inner.get_table_item(0) : nullptr;
      if (count != nullptr && count == outer.get_select_item(0).expr_
          && count->get_expr_type() == T_FUN_COUNT && !count->is_param_distinct()
          && count->get_real_param_count() == 0 && count->get_param_count() == 0
          && is_simple_block(inner) && !inner.has_group_by()
          && inner.get_select_item_size() == 1 && inner.get_condition_exprs().count() == 1
          && inner.get_match_exprs().count() == 1 && inner.get_order_item_size() == 1
          && inner.get_limit_expr() != nullptr && inner.get_limit_expr()->is_const_raw_expr()
          && !inner.get_limit_expr()->has_flag(CNT_DYNAMIC_PARAM)
          && inner.get_offset_expr() == nullptr && inner.get_limit_percent_expr() == nullptr
          && !inner.has_fetch() && !inner.is_fetch_with_ties()
          && base != nullptr && base->is_basic_table() && !base->is_view_table_
          && !base->is_system_table_ && !base->is_index_table_
          && base->cte_type_ == TableItem::NOT_CTE && base->exec_params_.empty()
          && base->sample_info_ == nullptr && base->snapshot_query_expr_ == nullptr
          && inner.get_from_item(0).table_id_ == base->table_id_) {
        const ObRawExpr *output = inner.get_select_item(0).expr_;
        const ObRawExpr *match_expr = inner.get_match_exprs().at(0);
        const ObRawExpr *filter = inner.get_condition_exprs().at(0);
        if (output != nullptr && output->is_column_ref_expr()
            && static_cast<const ObColumnRefRawExpr *>(output)->get_table_id() == base->table_id_
            && !static_cast<const ObColumnRefRawExpr *>(output)->is_generated_column()
            && match_expr != nullptr && match_expr->get_expr_class() == ObRawExpr::EXPR_MATCH_AGAINST
            && filter != nullptr
            && (filter == match_expr || (filter->get_expr_type() == T_OP_BOOL
                && filter->get_param_count() == 1 && filter->get_param_expr(0) == match_expr))
            && inner.get_order_item(0).expr_ == match_expr
            && inner.get_order_item(0).order_type_ == default_desc_direction()) {
          const ObMatchFunRawExpr *match = static_cast<const ObMatchFunRawExpr *>(match_expr);
          const ObRawExpr *key = match->get_search_key();
          valid = !match->is_es_match() && !match->get_match_columns().empty()
              && match->get_columns_boosts().empty() && match->get_param_text_expr() == nullptr
              && key != nullptr && key->is_const_raw_expr() && !key->has_flag(CNT_DYNAMIC_PARAM);
          for (int64_t i = 0; valid && i < match->get_match_columns().count(); ++i) {
            const ObRawExpr *column = match->get_match_columns().at(i);
            valid = column != nullptr && column->is_column_ref_expr()
                && static_cast<const ObColumnRefRawExpr *>(column)->get_table_id() == base->table_id_
                && !static_cast<const ObColumnRefRawExpr *>(column)->is_generated_column();
          }
        }
      }
    }
  }
  return valid;
}

int ObTransformSimplifyOrderby::transform_one_stmt(common::ObIArray<ObParentDMLStmt> &parent_stmts,
                                                   ObDMLStmt *&stmt,
                                                   bool &trans_happened)
{
  int ret = OB_SUCCESS;
  bool remove_duplicates = false;
  bool subquery_happened = false;
  bool view_happened = false;
  bool set_happened = false;
  bool force_serial_set_order = false;
  trans_happened = false;
  UNUSED(parent_stmts);
  if (OB_ISNULL(stmt) || OB_ISNULL(ctx_) || OB_ISNULL(ctx_->session_info_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ctx_->session_info_->is_serial_set_order_forced(force_serial_set_order))) {
  } else if (OB_FAIL(remove_order_by_for_subquery(stmt, subquery_happened))) {
  } else if (OB_FAIL(remove_order_by_for_view_stmt(stmt, view_happened, force_serial_set_order))) {
  } else if (OB_FAIL(remove_order_by_for_set_stmt(stmt, set_happened, force_serial_set_order))) {
  } else if (OB_FAIL(remove_order_by_duplicates(stmt, remove_duplicates))) {
  } else {
    trans_happened = (subquery_happened || view_happened || remove_duplicates || set_happened);
    OPT_TRACE("remove order by for subquery:", subquery_happened);
    OPT_TRACE("remove order by for view:", view_happened);
    OPT_TRACE("remove order by duplicates:", remove_duplicates);
    OPT_TRACE("remove order by for set stmt:", set_happened);
  }
  if (OB_SUCC(ret) && trans_happened) {
    if (OB_FAIL(add_transform_hint(*stmt))) {
    }
  }
  return ret;
}

// for select/update/delete/insert
// Remove order by clause from subquery
int ObTransformSimplifyOrderby::remove_order_by_for_subquery(ObDMLStmt *stmt, bool &trans_happened)
{
  int ret = OB_SUCCESS;
  trans_happened = false;
  if (OB_ISNULL(stmt)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ObIArray<ObQueryRefRawExpr*> &subquery_exprs = stmt->get_subquery_exprs();
    ObSelectStmt *subquery = NULL;
    bool happened = false;
    for (int64_t i = 0; OB_SUCC(ret) && i < subquery_exprs.count(); ++i) {
      ObQueryRefRawExpr *query_ref = subquery_exprs.at(i);
      if (OB_ISNULL(query_ref) || OB_ISNULL(subquery = query_ref->get_ref_stmt())) {
        ret = OB_ERR_UNEXPECTED;
      } else if (!subquery->has_limit() &&
                 !subquery->is_contains_assignment() &&
                 subquery->has_order_by() &&
                 OB_FAIL(do_remove_stmt_order_by(subquery, happened))) {
      } else {
        trans_happened |= happened;
      }
    }
  }
  return ret;
}

int ObTransformSimplifyOrderby::remove_order_by_for_view_stmt(ObDMLStmt *stmt, bool &trans_happened, bool &force_serial_set_order)
{
  int ret = OB_SUCCESS;
  bool can_remove = false;
  trans_happened = false;
  if (OB_ISNULL(stmt)) {
    ret = OB_ERR_UNEXPECTED;
  //select stmt cannot have window function,eg:
  // SELECT last_value(c1) OVER (PARTITION BY c2) FROM (SELECT * FROM t1 ORDER BY c1, c2)s
  // ORDER BY c1, c2;
  } else if (!stmt->is_sel_del_upd()) {
    /*do nothing*/
  } else if (stmt->is_select_stmt() && static_cast<ObSelectStmt*>(stmt)->has_window_function()) {
    /*do nothing*/
  } else {
    can_remove = true;
  }
  if (OB_SUCC(ret) && can_remove) {
    ObSelectStmt *view_stmt = NULL;
    ObSelectStmt *select_stmt = NULL;
    bool happened = false;
    ObIArray<TableItem *> &table_items = stmt->get_table_items();
    for (int64_t i = 0; OB_SUCC(ret) && i < table_items.count(); ++i) {
      const TableItem * table_item = table_items.at(i);
      if (OB_ISNULL(table_item)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (!table_item->is_generated_table() &&
                 !table_item->is_lateral_table()) {
        /*do nothing*/
      } else if (OB_ISNULL(view_stmt = table_item->ref_query_)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (view_stmt->is_contains_assignment() ||
                 (force_serial_set_order && view_stmt->is_set_stmt())) {
        // do nothing
      } else if (1 == table_items.count()) {
        if (stmt->is_select_stmt()) {
          select_stmt = static_cast<ObSelectStmt*>(stmt);
          if ((((select_stmt->has_group_by() || select_stmt->has_distinct() ||
                     select_stmt->has_order_by()) && !view_stmt->has_limit()) ||
                     is_count_over_limited_fts_view(*select_stmt)) &&
                     view_stmt->has_order_by() &&
                     OB_FAIL(do_remove_stmt_order_by(view_stmt, happened))) {
          } else {
            trans_happened |= happened;
          }
        }
      } else if (!view_stmt->has_limit() && view_stmt->has_order_by() &&
                 OB_FAIL(do_remove_stmt_order_by(view_stmt, happened))) {
      } else {
        trans_happened |= happened;
      }
    }
  }
  return ret;
}

int ObTransformSimplifyOrderby::do_remove_stmt_order_by(ObSelectStmt *select_stmt, bool &trans_happened)
{
  int ret = OB_SUCCESS;
  trans_happened = false;
  ObSEArray<OrderItem, 4> new_order_items;
  if (OB_ISNULL(select_stmt)) {
    ret = OB_ERR_UNEXPECTED;
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < select_stmt->get_order_item_size(); ++i) {
    const OrderItem &order_item = select_stmt->get_order_item(i);
    if (OB_ISNULL(order_item.expr_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (order_item.expr_->has_flag(CNT_SUB_QUERY)) {
      // in case the subquery returns more than one rows,
      // we should keep the subquery which may returns error
      if (OB_FAIL(new_order_items.push_back(order_item))) {
      }
    }
  }
  if (OB_SUCC(ret) && select_stmt->get_order_item_size() != new_order_items.count()) {
    trans_happened = true;
    if (OB_FAIL(select_stmt->get_order_items().assign(new_order_items))) {
    }
  }
  return ret;
}

int ObTransformSimplifyOrderby::remove_order_by_duplicates(ObDMLStmt *stmt,
                                                           bool &trans_happened)
{
  int ret = OB_SUCCESS;
  trans_happened = false;
  if (OB_ISNULL(stmt)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (!stmt->is_sel_del_upd() ||
             (stmt->is_select_stmt() && (static_cast<ObSelectStmt*>(stmt)->has_rollup()))) {
    //do nothing
  } else {
    ObIArray<OrderItem> &order_items = stmt->get_order_items();
    ObIArray<ObRawExpr*> &cond_exprs = stmt->get_condition_exprs();
    if (order_items.count() > 1) {
      ObArray<OrderItem> new_order_items;
      int64_t N = order_items.count();
      int64_t M = cond_exprs.count();
      bool is_exist = false;
      for (int64_t i = 0; OB_SUCC(ret) && i < N; ++i) {
        OrderItem &cur_item = order_items.at(i);
        is_exist = false;
        if (OB_FAIL(exist_item_by_expr(cur_item.expr_, new_order_items, is_exist))) {
        } else if (!is_exist) {
          for (int64_t j = 0; OB_SUCC(ret) && !is_exist && j < M; ++j) {
            ObRawExpr *op_expr = cond_exprs.at(j);
            if (OB_ISNULL(op_expr)) {
              ret = OB_ERR_UNEXPECTED;
            } else if (T_OP_EQ == op_expr->get_expr_type()) {
              if (op_expr->get_param_count() != 2) {
                ret = OB_ERR_UNEXPECTED;
              } else {
                ObRawExpr *left_param = op_expr->get_param_expr(0);
                ObRawExpr *right_param = op_expr->get_param_expr(1);
                bool is_consistent = false;
                if (OB_ISNULL(left_param) || OB_ISNULL(right_param)) {
                 ret = OB_ERR_UNEXPECTED;
                } else if (OB_FAIL(ObRelationalExprOperator::is_equal_transitive(
                                              left_param->get_result_type(),
                                              right_param->get_result_type(), is_consistent))) {
                } else if (!is_consistent) {
                  // do nothing
                } else if (right_param == cur_item.expr_) {
                  if (OB_FAIL(exist_item_by_expr(left_param, new_order_items, is_exist))) {
                  }
                } else if (left_param == cur_item.expr_) {
                  if (OB_FAIL(exist_item_by_expr(right_param, new_order_items, is_exist))) {
                  }
                }
              }
            } else {/* do nothing */}
          }
        } else {/*do nothing*/}
        if (OB_SUCC(ret) && !is_exist) {
          if (OB_FAIL(new_order_items.push_back(cur_item))) {
          }
        }
      }// for
      if (OB_SUCC(ret)) {
        if (N != new_order_items.count()) {
          trans_happened = true;
        }
        if (OB_FAIL(order_items.assign(new_order_items))) {
        }
      }
    }
  }
  return ret;
}

int ObTransformSimplifyOrderby::exist_item_by_expr(ObRawExpr *expr, ObIArray<OrderItem> &order_items, bool &is_exist)
{
  int ret = OB_SUCCESS;
  is_exist = false;
  int64_t N = order_items.count();
  if (OB_ISNULL(expr)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < N && !is_exist; ++i) {
      OrderItem& this_item = order_items.at(i);
      if (OB_ISNULL(this_item.expr_)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (this_item.expr_ == expr) {
        is_exist = true;
      } else if (this_item.expr_->same_as(*expr)) {
        is_exist = true;
      }
    }
  }
  return ret;
}

int ObTransformSimplifyOrderby::remove_order_by_for_set_stmt(ObDMLStmt *&stmt, bool &trans_happened, bool &force_serial_set_order)
{
  int ret = OB_SUCCESS;
  trans_happened = false;
  if (OB_ISNULL(stmt) || OB_ISNULL(ctx_) || OB_ISNULL(ctx_->session_info_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (!(stmt->is_select_stmt() && static_cast<ObSelectStmt*>(stmt)->is_set_stmt())) {
    //not set stmt, do nothing
  } else if (force_serial_set_order && 
             ObSelectStmt::UNION == static_cast<ObSelectStmt*>(stmt)->get_set_op() &&
             !static_cast<ObSelectStmt*>(stmt)->is_set_distinct()) {
    // When force_serial_set_order is set, UNION ALL keeps order items in child statements.
  } else {
    ObIArray<ObSelectStmt*> &child_stmts = static_cast<ObSelectStmt*>(stmt)->get_set_query();
    ObSelectStmt *child_stmt = NULL;
    for (int64_t i = 0; OB_SUCC(ret) && i < child_stmts.count(); ++i) {
      bool happened = false;
      if (OB_ISNULL(child_stmt = child_stmts.at(i))) {
        ret = OB_ERR_UNEXPECTED;
      } else if (child_stmt->is_contains_assignment()) {
        // do nothing
      } else if (!child_stmt->has_limit() &&
                 child_stmt->has_order_by() &&
                 OB_FAIL(do_remove_stmt_order_by(child_stmt, happened))) {
      } else {
        trans_happened |= happened;
      }
    }
  }
  return ret;
}
