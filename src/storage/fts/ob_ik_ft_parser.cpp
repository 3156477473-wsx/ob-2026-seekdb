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

#define USING_LOG_PREFIX STORAGE_FTS

#include "src/storage/fts/ob_ik_ft_parser.h"

#include "lib/charset/ob_charset.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/utility/ob_macro_utils.h"
#include "lib/utility/utility.h"
#include "data_plane/fts/ob_fts_struct.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "storage/fts/dict/ob_ft_dict.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_dict_hub.h"
#include "storage/fts/dict/ob_ft_range_dict.h"
#include "storage/fts/ik/ob_ik_arbitrator.h"
#include "storage/fts/ik/ob_ik_cjk_processor.h"
#include "storage/fts/ik/ob_ik_letter_processor.h"
#include "storage/fts/ik/ob_ik_processor.h"
#include "storage/fts/ik/ob_ik_quantifier_processor.h"
#include "storage/fts/ik/ob_ik_surrogate_processor.h"

namespace oceanbase
{
namespace storage
{
int ObIKFTParser::init(const ObFTParserParam &param)
{
  int ret = OB_SUCCESS;

  if (IS_INIT) {
    ret = OB_INIT_TWICE;
  } else {
    coll_type_ = ObCollationType::CS_TYPE_INVALID;
    if (OB_ISNULL(param.cs_) || OB_ISNULL(param.cs_->name)) {
      ret = OB_INVALID_ARGUMENT;
    } else if (CS_TYPE_INVALID == (coll_type_ = ObCharset::collation_type(param.cs_->name))) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_FAIL(init_dict(param))) {
    } else if (OB_FAIL(init_ctx(param))) {
    } else if (OB_FAIL(init_segmenter(param))) {
    }

    if (OB_FAIL(ret)) {
      reset();
    } else {
      is_inited_ = true;
    }
  }

  return ret;
}

int ObIKFTParser::reinit_ctx(const ObFTParserParam &param)
{
  int ret = OB_SUCCESS;
  if (!IS_INIT) {
    ret = OB_NOT_INIT;
  } else if (OB_ISNULL(param.cs_) || OB_ISNULL(param.cs_->name)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // 重置各 segmenter 内跨文档残留的状态（连续数字范围、未结算命中）。
    for (ObList<ObIIKProcessor *, ObIAllocator>::iterator iter = segmenters_.begin();
         OB_SUCC(ret) && iter != segmenters_.end();
         iter++) {
      if (OB_NOT_NULL(*iter)) {
        (*iter)->reset();
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_ISNULL(ctx_)) {
        if (OB_FAIL(init_ctx(param))) {
        }
      } else {
        // 原地复用 ctx 对象：只重绑文档与游标，不重新分配。
        ctx_->rebind(param.fulltext_, param.ft_length_);
      }
    }
  }
  return ret;
}

int ObIKFTParser::get_next_token(const char *&word,
                                 int64_t &word_len,
                                 int64_t &char_cnt,
                                 int64_t &word_freq)
{
  int ret = OB_SUCCESS;
  const char *output_word;
  int64_t len;
  int64_t offset;
  int64_t cnt;

  if (!IS_INIT) {
    ret = OB_NOT_INIT;
  } else {
    bool accept_token = false;
    while (OB_SUCC(ret) && !accept_token) {
      if (OB_FAIL(produce())) {
      } else if (OB_FAIL(ctx_->get_next_token(output_word, len, offset, cnt))) {
        if (OB_ITER_END == ret) {
          // ok, end this iter
        } else {
        }
      } else {
        bool is_stop = false;
        // if (!OB_ISNULL(dict_stop_)
        //     && OB_FAIL(dict_stop_->match(ObString(len, output_word + offset), is_stop))) {
        //   LOG_WARN("Failed to match stopwords", K(ret));
        // } else
        if (!is_stop) {
          word = output_word + offset;
          word_len = len;
          char_cnt = cnt;
          word_freq = 1;
          accept_token = true;
        } else {
        }
      }
    }
  }

  return ret;
}

int ObIKFTParser::produce()
{
  int ret = OB_SUCCESS;
  // Loop until end or has data to output
  while (OB_SUCC(ret) && ctx_->result_list().empty() && !ctx_->iter_end()) {
    if (OB_FAIL(process_next_batch())) {
      if (OB_ITER_END == ret) {
        // ok
      } else {
      }
    }
  }
  if (OB_ITER_END == ret) {
    ret = OB_SUCCESS;
  }
  return ret;
}

int ObIKFTParser::process_one_char(TokenizeContext &ctx,
                                   const char *ch,
                                   const uint8_t char_len,
                                   const ObFTCharUtil::CharType type)
{
  int ret = OB_SUCCESS;
  // proces by char with all segmenters
  for (ObList<ObIIKProcessor *, ObIAllocator>::iterator iter = segmenters_.begin();
       OB_SUCC(ret) && iter != segmenters_.end();
       iter++) {
    if (OB_FAIL((*iter)->process(ctx))) {
    }
  }
  return ret;
}

int ObIKFTParser::process_next_batch()
{
  int ret = OB_SUCCESS;
  ctx_->reset_resource();

  // handle next segmenter
  bool do_seg = false;

  if (ctx_->iter_end()) {
    ret = OB_ITER_END;
  } else {
    while (OB_SUCC(ret) && !do_seg && !ctx_->iter_end()) {
      const char *ch;
      uint8_t char_len = 0;
      ObFTCharUtil::CharType type = ObFTCharUtil::CharType::USELESS;
      if (OB_FAIL(ctx_->current_char(ch, char_len))) {
      } else if (OB_FAIL(ctx_->current_char_type(type))) {
      } else if (OB_FAIL(process_one_char(*ctx_, ch, char_len, type))) {
      } else {
        // 1. check segmention
        if (ctx_->handle_size() > SEGMENT_LIMIT && type == ObFTCharUtil::CharType::USELESS) {
          do_seg = true;
        }

        // 2. move to next;
        if (OB_FAIL(ctx_->step_next())) {
          if (OB_ITER_END == ret) {
          } else {
          }
        }
      } // end of one batch
    }

    if (OB_SUCC(ret) || OB_ITER_END == ret) {
      ObIKArbitrator arb;
      if (OB_FAIL(arb.process(*ctx_))) {
      } else if (OB_FAIL(arb.output_result(*ctx_))) {
      }
    } else {
      // Already logged.
    }
  }

  return ret;
}

namespace
{
// 线程级 IK 解析器缓存。
//
// TOKENIZE() 这类路径会以完全相同的解析器参数被反复调用（评测里是数千次），
// 而每次完整 init() 的固定开销（借词典视图、构建 4 个 segmenter、建 ctx）
// 远大于短文本本身的分词开销。这里为每个线程保留一个解析器及其专用 arena：
// 首次建好后，后续调用只 rebind ctx。
//
// 安全性：
//  - 解析器与 arena 同生命周期，不依赖调用方的临时分配器，不会悬垂；
//  - 同一线程若出现嵌套使用（缓存被占用），自动回退到普通新建路径；
//  - 一旦复用出错就永久停用缓存，全部走原路径；
//  - arena 超过上限时丢弃缓存重建，保证内存有界。
struct ObIKParserCache
{
  static constexpr int64_t MAX_ARENA_BYTES = 8LL * 1024 * 1024;
  common::ObArenaAllocator arena_;
  ObIKFTParser *parser_;
  ObCollationType coll_type_;
  ObFTIKParam::Mode mode_;
  bool in_use_;
  bool available_;

  ObIKParserCache()
    : arena_(lib::ObLabel("IKParserCache")),
      parser_(nullptr),
      coll_type_(ObCollationType::CS_TYPE_INVALID),
      mode_(ObFTIKParam::Mode::SMART),
      in_use_(false),
      available_(true)
  {}
  ~ObIKParserCache() { parser_ = nullptr; }

  void drop()
  {
    // 丢弃缓存：先断开引用再回收 arena，避免 arena 复用后旧对象被访问。
    parser_ = nullptr;
    coll_type_ = ObCollationType::CS_TYPE_INVALID;
    mode_ = ObFTIKParam::Mode::SMART;
    in_use_ = false;
    arena_.reset();
  }
};

ObIKParserCache &ik_parser_cache()
{
  static thread_local ObIKParserCache cache;
  return cache;
}
} // namespace

int ObIKFTParserDesc::segment(ObFTParserParam *param, ObITokenIterator *&iter) const
{
  int ret = OB_SUCCESS;
  ObIKFTParser *parser = nullptr;
  ObFTDictHub *hub = nullptr;
  ObIKParserCache &cache = ik_parser_cache();

  if (OB_ISNULL(param) || OB_UNLIKELY(!param->is_valid())) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    if (cache.available_ && cache.arena_.total() > ObIKParserCache::MAX_ARENA_BYTES) {
      cache.drop();
    }
    const ObCollationType wanted = ObCharset::collation_type(param->cs_->name);
    const ObFTIKParam::Mode wanted_mode = param->ik_param_.mode_;
    bool reused = false;
    if (cache.available_ && !cache.in_use_ && OB_NOT_NULL(cache.parser_)
        && cache.coll_type_ == wanted && cache.mode_ == wanted_mode) {
      if (OB_SUCC(cache.parser_->reinit_ctx(*param))) {
        parser = cache.parser_;
        cache.in_use_ = true;
        reused = true;
        iter = parser;
      } else {
        // 复用异常：停用缓存，本线程后续全部走普通路径。
        cache.available_ = false;
        cache.drop();
        ret = OB_SUCCESS;
      }
    }

    if (OB_SUCC(ret) && !reused) {
      const bool can_cache = cache.available_ && !cache.in_use_ && OB_ISNULL(cache.parser_);
      common::ObIAllocator *alloc = can_cache ? &cache.arena_ : param->allocator_;
      if (OB_FAIL(ObFTParseData::instance().get_dict_hub(hub))) {
      } else if (OB_ISNULL(parser = OB_NEWx(ObIKFTParser, alloc, *alloc, hub))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(parser->init(*param))) {
        OB_DELETEx(ObIKFTParser, alloc, parser);
        parser = nullptr;
      } else {
        iter = parser;
        if (can_cache) {
          cache.parser_ = parser;
          cache.coll_type_ = wanted;
          cache.mode_ = wanted_mode;
          cache.in_use_ = true;
        }
      }
    }
  }

  return ret;
}

void ObIKFTParserDesc::free_token_iter(ObFTParserParam *param,
                                       ObITokenIterator *&iter) const
{
  if (OB_ISNULL(iter)) {
    // nothing to do
  } else if (iter == static_cast<ObITokenIterator *>(ik_parser_cache().parser_)) {
    // 缓存成员：只解除占用，不销毁。
    ik_parser_cache().in_use_ = false;
    iter = nullptr;
  } else {
    iter->~ObITokenIterator();
    param->allocator_->free(iter);
  }
}


int ObIKFTParserDesc::get_add_word_flag(ObAddWordFlag &flag) const
{
  int ret = OB_SUCCESS;
  flag.set_casedown();
  flag.set_groupby_word();
  return ret;
}

int ObIKFTParser::init_dict(const ObFTParserParam &param)
{
  int ret = OB_SUCCESS;
  ObFTDictDesc main_desc("main_dict", ObFTDictType::DICT_IK_MAIN,
                        CHARSET_UTF8MB4, CS_TYPE_UTF8MB4_BIN);
  ObFTDictDesc quan_desc("quan_dict", ObFTDictType::DICT_IK_QUAN,
                        CHARSET_UTF8MB4, CS_TYPE_UTF8MB4_BIN);
  if (OB_ISNULL(hub_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (should_read_newest_table()) {
    // Existing dormant reload path; no mutable-generation cache is introduced.
  } else if (OB_FAIL(hub_->borrow_dict_view(main_desc, dict_main_))) {
  } else if (OB_ISNULL(dict_main_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(hub_->borrow_dict_view(quan_desc, dict_quan_))) {
  } else if (OB_ISNULL(dict_quan_)) {
    ret = OB_ERR_UNEXPECTED;
  }
  // Preserve42's avoidance of the inactive stopword range view.
  return ret;
}



int ObIKFTParser::init_ctx(const ObFTParserParam &param)
{
  int ret = OB_SUCCESS;

  if (coll_type_ == common::CS_TYPE_INVALID) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(ctx_ = OB_NEWx(TokenizeContext,
                                      &allocator_,
                                      coll_type_,
                                      allocator_,
                                      param.fulltext_,
                                      param.ft_length_,
                                      param.ik_param_.mode_ == ObFTIKParam::Mode::SMART))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_FAIL(ctx_->init())) {
  }
  if (OB_FAIL(ret)) {
    OB_DELETEx(TokenizeContext, &allocator_, ctx_);
  }
  return ret;
}

int ObIKFTParser::init_segmenter(const ObFTParserParam &param)
{
  int ret = OB_SUCCESS;
  // do have an order
  ObIKLetterProcessor *letter_seg = nullptr;
  ObIKQuantifierProcessor *cnqsg = nullptr;
  ObIKCJKProcessor *cjksg = nullptr;
  ObIKSurrogateProcessor *surrogate_seg = nullptr;
  if (OB_ISNULL(letter_seg = OB_NEWx(ObIKLetterProcessor, &allocator_))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(dict_quan_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_ISNULL(cnqsg = OB_NEWx(ObIKQuantifierProcessor, &allocator_, *dict_quan_, allocator_))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(dict_main_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_ISNULL(cjksg = OB_NEWx(ObIKCJKProcessor, &allocator_, *dict_main_, allocator_))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(surrogate_seg = OB_NEWx(ObIKSurrogateProcessor, &allocator_))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_FAIL(segmenters_.push_back(letter_seg))) {
  } else if (FALSE_IT(letter_seg = nullptr)) {
  } else if (OB_FAIL(segmenters_.push_back(cnqsg))) {
  } else if (FALSE_IT(cnqsg = nullptr)) {
  } else if (OB_FAIL(segmenters_.push_back(cjksg))) {
  } else if (FALSE_IT(cjksg = nullptr)) {
  } else if (OB_FAIL(segmenters_.push_back(surrogate_seg))) {
  } else if (OB_FALSE_IT(surrogate_seg = nullptr)) {
  }
  // push back by order, quantifier is before cjk

  if (OB_FAIL(ret)) {
    OB_DELETEx(ObIKLetterProcessor, &allocator_, letter_seg);
    OB_DELETEx(ObIKQuantifierProcessor, &allocator_, cnqsg);
    OB_DELETEx(ObIKCJKProcessor, &allocator_, cjksg);
    OB_DELETEx(ObIKSurrogateProcessor, &allocator_, surrogate_seg);
  }
  return ret;
}

void ObIKFTParser::reset()
{
  if (!OB_ISNULL(ctx_)) {
    ctx_->~TokenizeContext();
    allocator_.free(ctx_);
    ctx_ = nullptr;
  }

  for (ObIIKProcessor *segmenter : segmenters_) {
    if (!OB_ISNULL(segmenter)) {
      segmenter->~ObIIKProcessor();
      allocator_.free(segmenter);
    }
  }
  segmenters_.clear();

  // The dict views are owned by the dict hub, so they are only borrowed here.
  dict_main_ = nullptr;
  dict_quan_ = nullptr;

  is_inited_ = false;
}

bool ObIKFTParser::should_read_newest_table() const { return false; }


} //  namespace storage
} //  namespace oceanbase
