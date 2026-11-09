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

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"
#include "sql/engine/ob_exec_context.h"

namespace oceanbase
{
namespace storage
{
namespace
{
int compare_owned_binary_doc_ids(const ObDatum &left, const ObDatum &right,
    int &cmp, const common::ObDatumAccessContext *access_ctx)
{
  int ret = OB_SUCCESS;
  // pack_ == 16 proves length 16, non-NULL, and FlagType::NONE together.
  // This callback is installed only for the actual VARCHAR/BINARY ID domain.
  // Its operands are the merger's owned ObDocIdExt values (40-byte storage).
  if (OB_LIKELY(16 == left.pack_ && 16 == right.pack_
                && nullptr != left.ptr_ && nullptr != right.ptr_)) {
    const int result = MEMCMP(left.ptr_, right.ptr_, 16);
    cmp = result > 0 ? 1 : (result < 0 ? -1 : 0);
  } else {
    // Keep the exact original nullable/string comparator for every other datum.
    const common::ObDatumBasicFuncs *funcs = ObDatumFuncs::get_basic_func(
        common::ObVarcharType, common::CS_TYPE_BINARY);
    ret = funcs->null_first_cmp_(left, right, cmp, access_ctx);
  }
  return ret;
}
} // namespace

int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (nullptr != param.base_param_->relevance_expr_
      && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
    minimum_should_match_ = param.minimum_should_match_;
    bulk_union_enabled_ = can_use_bulk_union();
    if (bulk_union_enabled_) {
      const sql::ObDatumMeta &meta = iter_param_->id_proj_expr_->datum_meta_;
      bulk_union_cmp_ = ObDatumFuncs::get_basic_func(meta.type_, meta.cs_type_)->null_first_cmp_;
      // The base merger has already validated the same comparator.
      bulk_union_enabled_ = nullptr != bulk_union_cmp_;
      if (bulk_union_enabled_ && common::ObVarcharType == meta.type_
          && common::CS_TYPE_BINARY == meta.cs_type_) {
        bulk_union_cmp_ = compare_owned_binary_doc_ids;
      }
    }
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  bulk_union_enabled_ = false;
  bulk_union_cmp_ = nullptr;
  minimum_should_match_ = -1;
  for (int64_t i = 0; i < BULK_UNION_MAX_FANOUT; ++i) {
    bulk_union_eof_[i] = false;
  }
  bm25_param_estimator_.reset();
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRDaaTIterImpl::reuse(switch_tablet);
  for (int64_t i = 0; i < BULK_UNION_MAX_FANOUT; ++i) {
    bulk_union_eof_[i] = false;
  }
  // Recheck only at the rescan boundary, before any posting is read.
  bulk_union_enabled_ = nullptr != bulk_union_cmp_ && can_use_bulk_union();
}


bool ObTextDaaTIter::can_use_bulk_union() const
{
  bool eligible = is_inited_ && nullptr != iter_param_ && nullptr != dim_iters_
      && sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE == mode_flag_
      && !function_lookup_mode_ && minimum_should_match_ >= 0
      && dim_iters_->count() > 0 && dim_iters_->count() <= BULK_UNION_MAX_FANOUT
      && minimum_should_match_ <= dim_iters_->count()
      && nullptr == iter_param_->relevance_expr_
      && !iter_param_->need_project_relevance() && !iter_param_->need_filter()
      && !iter_param_->need_pushdown_topk();
  for (int64_t i = 0; eligible && i < dim_iters_->count(); ++i) {
    // The text merger already owns this concrete type in reset()/reuse().
    const ObTextRetrievalDaaTTokenIter *iter =
        static_cast<const ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
    eligible = nullptr != iter && iter->supports_owned_doc_id_span();
  }
  return eligible;
}

int ObTextDaaTIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (!bulk_union_enabled_) {
    ret = ObSRDaaTIterImpl::get_next_rows(capacity, count);
  } else {
    count = 0;
    if (IS_NOT_INIT) {
      ret = OB_NOT_INIT;
    } else if (0 == capacity) {
    } else if (iter_param_->limit_param_->is_valid()
        && output_row_cnt_ >= iter_param_->limit_param_->limit_) {
      ret = OB_ITER_END;
    } else if (OB_FAIL(pre_process())) {
    } else {
      const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
      ret = merge_owned_doc_id_spans(real_capacity, count);
      if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
        // Preserve the base convention: a partial final batch returns SUCCESS;
        // the following call returns ITER_END with zero rows.
        ObExpr *id_expr = iter_param_->id_proj_expr_;
        ObEvalCtx *eval_ctx = iter_param_->eval_ctx_;
        if (eval_ctx->is_vectorized() && id_expr->is_batch_result()
            && common::ObVarcharType == id_expr->datum_meta_.type_
            && common::CS_TYPE_BINARY == id_expr->datum_meta_.cs_type_) {
          // This domain uses set_datum_shallow: each datum is fully replaced,
          // so resetting its reserved result pointer per row is unnecessary.
          ObDatum *datums = id_expr->locate_batch_datums(*eval_ctx);
          for (int64_t i = 0; i < count; ++i) {
            datums[i].set_datum(buffered_domain_ids_[i].get_datum());
          }
          id_expr->get_evaluated_flags(*eval_ctx).set_all(count);
          id_expr->set_evaluated_projected(*eval_ctx);
          ret = OB_SUCCESS;
        } else {
          ret = ObSRDaaTIterImpl::project_results(count);
        }
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::try_count_all(int64_t &total, bool &supported)
{
  int ret = OB_SUCCESS;
  total = 0;
  supported = false;
  if (!bulk_union_enabled_ || !can_use_bulk_union() || nullptr == iter_param_->limit_param_
      || iter_param_->limit_param_->is_valid() || iter_param_->limit_param_->offset_ != 0) return ret;
  supported = true; // All qualification precedes the first cursor advance.
  if (OB_FAIL(pre_process())) return ret;
  while (OB_SUCC(ret)) {
    int64_t rows = 0;
    if (OB_FAIL(iter_param_->eval_ctx_->exec_ctx_.fast_check_status(0))) break;
    const int next = merge_owned_doc_id_spans(MAX(iter_param_->max_batch_size_, int64_t(1)), rows, true);
    if (next != OB_SUCCESS && next != OB_ITER_END) {
      ret = next;
    } else if (rows < 0) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      total += rows;
      if (next == OB_ITER_END) break;
    }
  }
  return ret;
}

int ObTextDaaTIter::find_bulk_prefix(const ObDocIdExt *ids, const int64_t size,
    const ObDatum *upper_bound, int64_t &prefix) const
{
  int ret = OB_SUCCESS;
  prefix = size;
  if (nullptr != upper_bound) {
    // ids[0] is already strictly below the second head. Gallop before binary
    // search so an interleaved single-row winner costs only one extra compare.
    int64_t left = 1;
    int64_t right = size;
    int64_t probe = 1;
    int cmp = 0;
    while (OB_SUCC(ret) && probe < size) {
      if (OB_FAIL(bulk_union_cmp_(ids[probe].get_datum(), *upper_bound, cmp, nullptr))) {
      } else if (cmp >= 0) {
        right = probe;
        break;
      } else {
        left = probe + 1;
        probe = probe < size / 2 ? probe * 2 : size;
      }
    }
    while (OB_SUCC(ret) && left < right) {
      const int64_t mid = left + (right - left) / 2;
      if (OB_FAIL(bulk_union_cmp_(ids[mid].get_datum(), *upper_bound, cmp, nullptr))) {
      } else if (cmp < 0) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    prefix = left;
  }
  return ret;
}

int ObTextDaaTIter::merge_owned_doc_id_spans(const int64_t capacity, int64_t &count, const bool count_only)
{
  int ret = OB_SUCCESS;
  const int64_t fanout = dim_iters_->count();
  ObTextRetrievalDaaTTokenIter *iters[BULK_UNION_MAX_FANOUT] = {};
  const ObDocIdExt *spans[BULK_UNION_MAX_FANOUT] = {};
  int64_t sizes[BULK_UNION_MAX_FANOUT] = {};
  for (int64_t i = 0; i < fanout; ++i) {
    iters[i] = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
  }
  // These views live only for this call. Each child owns its fixed ID buffer;
  // another child's refill cannot invalidate an unconsumed view.
  while (OB_SUCC(ret) && count < capacity) {
    int64_t winner = -1;
    int64_t matched = 0;
    uint64_t matched_mask = 0;
    int64_t second = -1;
    int cmp = 0;
    for (int64_t i = 0; OB_SUCC(ret) && i < fanout; ++i) {
      ObTextRetrievalDaaTTokenIter *iter = iters[i];
      if (!bulk_union_eof_[i] && 0 == sizes[i]) {
        iter->get_owned_doc_id_span(spans[i], sizes[i]);
        if (0 == sizes[i]) {
          // No get_next_row() while a buffered head is still unconsumed.
          if (OB_FAIL(iter->get_next_row())) {
            if (OB_ITER_END == ret) {
              bulk_union_eof_[i] = true;
              ret = OB_SUCCESS;
            }
          } else {
            iter->get_owned_doc_id_span(spans[i], sizes[i]);
            if (OB_UNLIKELY(0 == sizes[i])) {
              ret = OB_ARRAY_OUT_OF_RANGE;
            }
          }
        }
      }
      if (OB_SUCC(ret) && sizes[i] > 0) {
        if (winner < 0) {
          winner = i;
          matched = 1;
          matched_mask = uint64_t(1) << i;
        } else if (OB_FAIL(bulk_union_cmp_(spans[i][0].get_datum(),
                                          spans[winner][0].get_datum(), cmp, nullptr))) {
        } else if (cmp < 0) {
          // The previous minimum is the new second distinct head.
          second = winner;
          winner = i;
          matched = 1;
          matched_mask = uint64_t(1) << i;
        } else if (0 == cmp) {
          ++matched;
          matched_mask |= uint64_t(1) << i;
        } else if (second < 0) {
          second = i;
        } else if (OB_FAIL(bulk_union_cmp_(spans[i][0].get_datum(),
                                          spans[second][0].get_datum(), cmp, nullptr))) {
        } else if (cmp < 0) {
          second = i;
        }
      }
    }
    if (OB_FAIL(ret)) {
    } else if (winner < 0) {
      ret = OB_ITER_END;
    } else {
      if (OB_SUCC(ret) && 1 == matched) {
        int64_t prefix = 0;
        int64_t search_size = sizes[winner];
        if (minimum_should_match_ <= 1) {
          // Do not compare IDs beyond what this call can consume, especially
          // for LIMIT 1. Add remaining OFFSET without overflowing the span size.
          search_size = MIN(search_size, capacity - count);
          const common::ObLimitParam &limit = *iter_param_->limit_param_;
          if (limit.is_valid()) {
            search_size = MIN(search_size, limit.limit_ - output_row_cnt_);
            if (input_row_cnt_ < limit.offset_) {
              search_size += MIN(sizes[winner] - search_size, limit.offset_ - input_row_cnt_);
            }
          }
        }
        const ObDatum *upper_bound = second < 0 ? nullptr : &spans[second][0].get_datum();
        if (OB_FAIL(find_bulk_prefix(spans[winner], search_size, upper_bound, prefix))) {
        } else {
          int64_t consumed = 0;
          if (minimum_should_match_ > 1) {
            // Every ID in this strictly bounded prefix matches only this child.
            consumed = prefix;
          } else if (count_only) {
            consumed = prefix;
            input_row_cnt_ += prefix;
            output_row_cnt_ += prefix;
            count += prefix;
          } else {
            const common::ObLimitParam &limit = *iter_param_->limit_param_;
            const int64_t skipped = limit.is_valid() && input_row_cnt_ < limit.offset_
                ? MIN(prefix, limit.offset_ - input_row_cnt_) : 0;
            int64_t emitted = MIN(prefix - skipped, capacity - count);
            if (limit.is_valid()) {
              emitted = MIN(emitted, limit.limit_ - output_row_cnt_);
            }
            if (OB_LIKELY(count >= 0 && emitted >= 0
                && count <= buffered_domain_ids_.count()
                && count <= buffered_relevances_.count()
                && emitted <= buffered_domain_ids_.count() - count
                && emitted <= buffered_relevances_.count() - count)) {
              // Assignment copies the complete owned 40-byte ID buffer and
              // rebinds its datum to the destination before the child advances.
              for (int64_t i = 0; i < emitted; ++i) {
                buffered_domain_ids_[count + i] = spans[winner][skipped + i];
              }
              if (emitted > 0) {
                MEMSET(&buffered_relevances_[count], 0, emitted * sizeof(double));
              }
              consumed = skipped + emitted;
              input_row_cnt_ += consumed;
              output_row_cnt_ += emitted;
              count += emitted;
              if (limit.is_valid() && emitted > 0 && output_row_cnt_ >= limit.limit_) {
                ret = OB_ITER_END;
              }
            } else {
              // Preserve the original partial progress on an invalid buffer.
              while (OB_SUCC(ret) && consumed < prefix && count < capacity) {
                ret = ObSRDaaTIterImpl::cache_result(count, spans[winner][consumed].get_datum(), 0.0);
                ++consumed;
              }
            }
          }
          // Every output ID is owned before the borrowed span advances.
          const int consume_ret = iters[winner]->consume_owned_doc_id_prefix(consumed);
          if (OB_SUCCESS != consume_ret) {
            ret = consume_ret;
          } else {
            // Mirror the actual child cursor, including OFFSET/LIMIT prefixes.
            spans[winner] += consumed;
            sizes[winner] -= consumed;
          }
        }
      } else if (OB_SUCC(ret)) {
        if (matched >= minimum_should_match_) {
          if (count_only) {
            ++input_row_cnt_;
            ++output_row_cnt_;
            ++count;
          } else {
            ret = ObSRDaaTIterImpl::cache_result(count, spans[winner][0].get_datum(), 0.0);
          }
        }
        // Consume every equal head, even if minimum_should_match rejects the ID
        // or cache_result reaches LIMIT. No borrowed pointer survives a reload.
        for (int64_t i = 0; i < fanout; ++i) {
          if (0 != (matched_mask & (uint64_t(1) << i))) {
            const int consume_ret = iters[i]->consume_owned_doc_id_prefix(1);
            if (OB_SUCCESS != consume_ret) {
              ret = consume_ret;
            } else {
              ++spans[i];
              --sizes[i];
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (iter_param_->need_project_relevance()) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
