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

#ifndef OB_BENG_FT_PARSER_H_
#define OB_BENG_FT_PARSER_H_

#include "lib/utility/ob_macro_utils.h"
#include "lib/utility/ob_print_utils.h"
#include "lib/ob_define.h"
#include "share/text_analysis/ob_text_analyzer.h"
#include "data_plane/fts/ob_fts_parser.h"

namespace oceanbase
{
namespace storage
{

class ObBEngFTParser final : public ObITokenIterator
{
public:
  static const int64_t FT_MIN_WORD_LEN = 3;
  static const int64_t FT_MAX_WORD_LEN = 84;
  // token 拷贝缓冲区的最小容量。BENG 的单文档 token 总量一般远小于这个值，
  // 因此多数文档只需一次分配。
  static const int64_t TOKEN_BUF_MIN_CAPACITY = 4096;
public:
  explicit ObBEngFTParser(common::ObIAllocator &allocator)
    : allocator_(allocator),
      analysis_ctx_(),
      english_analyzer_(),
      doc_(),
      token_stream_(nullptr),
      token_buf_(nullptr),
      token_buf_capacity_(0),
      token_buf_used_(0),
      is_inited_(false)
  {}
  ~ObBEngFTParser();

  int init(ObFTParserParam *param);
  void reset();
  virtual int get_next_token(
      const char *&word,
      int64_t &word_len,
      int64_t &char_len,
      int64_t &word_freq) override;

  VIRTUAL_TO_STRING_KV(K_(analysis_ctx), K_(english_analyzer), KP_(token_stream), K_(is_inited));
private:
  int segment(
      const common::ObDatum &doc,
      share::ObITokenStream *&token_stream);
  // 确保 token 缓冲区能容纳当前已用长度再加上 need_len 字节。
  int ensure_token_buf(const int64_t need_len);
private:
  common::ObIAllocator &allocator_;
  share::ObTextAnalysisCtx analysis_ctx_;
  share::ObEnglishTextAnalyzer english_analyzer_;
  common::ObDatum doc_;
  share::ObITokenStream *token_stream_;
  // token 拷贝缓冲区。原先每个 token 都做一次 allocator_.alloc()，
  // 这里改成按需增长的连续缓冲区，每次只从里面切分，减少小块分配。
  char *token_buf_;
  int64_t token_buf_capacity_;
  int64_t token_buf_used_;
  bool is_inited_;

  DISALLOW_COPY_AND_ASSIGN(ObBEngFTParser);
};

class ObBasicEnglishFTParserDesc final : public ObIFTParserDesc
{
public:
  ObBasicEnglishFTParserDesc();
  virtual ~ObBasicEnglishFTParserDesc() = default;
  virtual int segment(ObFTParserParam *param, ObITokenIterator *&iter) const override;
  virtual void free_token_iter(ObFTParserParam *param, ObITokenIterator *&iter) const override;
  virtual int get_add_word_flag(ObAddWordFlag &flag) const override;
};

} // end namespace storage
} // end namespace oceanbase

#endif // OB_BENG_FT_PARSER_H_
