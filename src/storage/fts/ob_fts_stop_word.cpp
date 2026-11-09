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

#include "object/ob_object.h"
#define USING_LOG_PREFIX STORAGE_FTS

#include "share/rc/ob_server_runtime.h"
#include "storage/fts/ob_fts_stop_word.h"
#include "storage/fts/ob_fts_parser_helper.h"
#include "storage/fts/ob_fts_parser_property.h"

namespace oceanbase
{
namespace storage
{

namespace {
// Product data only: derive exact packed keys from the same immutable builtin
// list that initializes StopWordSet. No SQL text, parser state or result cache.
class BuiltinAsciiStopWords final
{
public:
  BuiltinAsciiStopWords() : keys_(), count_(0), max_len_(0), valid_(true)
  {
    for (int64_t i = 0; valid_ && i < WORD_COUNT; ++i) {
      const char *word = ob_stop_word_list[i];
      int64_t len = 0;
      while (len < FTS_STOP_WORD_MAX_LENGTH && word[len] != '\0') ++len;
      if (len == 0 || len > 7 || len == FTS_STOP_WORD_MAX_LENGTH) {
        valid_ = false;
      } else {
        uint64_t key = len;
        for (int64_t j = 0; valid_ && j < len; ++j) {
          const unsigned char c = static_cast<unsigned char>(word[j]);
          valid_ = c >= 'a' && c <= 'z';
          key = (key << 8) | c;
        }
        if (valid_) {
          int64_t pos = 0;
          while (pos < count_ && keys_[pos] < key) ++pos;
          if (pos == count_ || keys_[pos] != key) {
            for (int64_t j = count_; j > pos; --j) keys_[j] = keys_[j - 1];
            keys_[pos] = key;
            ++count_;
          }
          max_len_ = std::max(max_len_, len);
        }
      }
    }
  }
  bool valid() const { return valid_ && count_ > 0; }
  int64_t size() const { return count_; }
  // Return true only when this exact ASCII domain was handled. Every other
  // input continues through charset conversion and the authoritative set.
  bool try_lookup(const common::ObString &word, bool &found) const
  {
    constexpr int64_t ASCII_SCAN_LIMIT = 64;
    if (!valid() || word.empty() || word.length() > ASCII_SCAN_LIMIT) return false;
    uint64_t key = word.length();
    for (int64_t i = 0; i < word.length(); ++i) {
      const unsigned char c = static_cast<unsigned char>(word.ptr()[i]);
      if (c < 'a' || c > 'z') return false;
      if (word.length() <= max_len_) key = (key << 8) | c;
    }
    found = false;
    if (word.length() <= max_len_) {
      int64_t lo = 0, hi = count_;
      while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (keys_[mid] < key) lo = mid + 1;
        else hi = mid;
      }
      found = lo < count_ && keys_[lo] == key;
    }
    return true;
  }
private:
  static constexpr int64_t WORD_COUNT = sizeof(ob_stop_word_list) / sizeof(ob_stop_word_list[0]);
  uint64_t keys_[WORD_COUNT];
  int64_t count_;
  int64_t max_len_;
  bool valid_;
};
const BuiltinAsciiStopWords builtin_ascii_stopwords;
} // namespace

////////////////////////////////////////////////////////////////////////////////
// class ObStopWordChecker
ObStopWordChecker::~ObStopWordChecker()
{
  destroy();
}

int ObStopWordChecker::init()
{
  int ret = OB_SUCCESS;
  
  

  if (inited_) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(stopword_set_.create(DEFAULT_STOPWORD_BUCKET_NUM, "StopWordSet", "StopWordSet"))) {
  } else {
    ObObjMeta stop_meta;
    stop_meta.set_varchar();
    stop_meta.set_collation_type(ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI);

    stopword_type_.set_meta(stop_meta);
    const int64_t stopword_count = sizeof(ob_stop_word_list) / sizeof(ob_stop_word_list[0]);
    for (int64_t i = 0; OB_SUCC(ret) && i < stopword_count; ++i) {
      ObFTWord stopword(STRLEN(ob_stop_word_list[i]), ob_stop_word_list[i], stopword_type_);
      if (OB_FAIL(stopword_set_.set_refactored(stopword))) {
      }
    }

    if (OB_SUCC(ret)) {
      inited_ = true;
      // This flag is published only by the initializer that inserted every
      // builtin word; a different/incomplete set never receives the shortcut.
      builtin_ascii_fast_path_ = builtin_ascii_stopwords.valid()
          && stopword_type_.get_collation_type() == common::CS_TYPE_UTF8MB4_GENERAL_CI
          && stopword_set_.size() == builtin_ascii_stopwords.size();
    }
  }
  return ret;
}

void ObStopWordChecker::destroy()
{
  if (inited_) {
    stopword_set_.destroy();
    inited_ = false;
  }
  builtin_ascii_fast_path_ = false;
}

int ObStopWordChecker::check_stopword(const ObFTWord &word, bool &is_stopword)
{
  int ret = OB_SUCCESS;
  
  
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(word.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (builtin_ascii_fast_path_ && !word.get_word().is_null()
      && word.get_word().flag_ == common::ObDatum::NONE
      && (word.get_collation_type() == common::CS_TYPE_UTF8MB4_GENERAL_CI
          || word.get_collation_type() == common::CS_TYPE_UTF8MB4_BIN
          || word.get_collation_type() == common::CS_TYPE_BINARY)
      && builtin_ascii_stopwords.try_lookup(word.get_word().get_string(), is_stopword)) {
    // Exact lowercase ASCII membership in the existing builtin set.
  } else {
    common::ObArenaAllocator allocator(lib::ObMemAttr("ChkStopWord"));
    common::ObString cmp_str;
    // do nothing set out with in if type is the same.
    if (OB_FAIL(common::ObCharset::charset_convert(
                                       allocator,
                                       word.get_word().get_string(),
                                       word.get_collation_type(),
                                       stopword_type_.get_collation_type(),
                                       cmp_str))) {
    } else {
      ObFTWord converted(cmp_str.length(), cmp_str.ptr(), stopword_type_);
      ret = stopword_set_.exist_refactored(converted);
      if (OB_HASH_NOT_EXIST == ret) {
        is_stopword = false;
        ret = OB_SUCCESS;
      } else if (OB_HASH_EXIST == ret) {
        is_stopword = true;
        ret = OB_SUCCESS;
      } else if (OB_SUCC(ret)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
      }
    }
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// class ObAddWord
ObAddWord::ObAddWord(
    const ObFTParserProperty &property,
    const ObObjMeta &meta,
    const ObAddWordFlag &flag,
    common::ObIAllocator &allocator,
    ObFTWordMap &word_map)
  : word_meta_(meta),
    allocator_(allocator),
    word_map_(word_map),
    min_max_word_cnt_(0),
    non_stopword_cnt_(0),
    stopword_cnt_(0),
    min_token_size_(property.min_token_size_),
    max_token_size_(property.max_token_size_),
    flag_(flag)
{
}

int ObAddWord::process_word(
    const char *word,
    const int64_t word_len,
    const int64_t char_cnt,
    const int64_t word_freq)
{
  int ret = OB_SUCCESS;
  bool is_stopword = false;
  ObFTWord src_word(word_len, word, word_meta_);
  ObFTWord dst_word;
  if (OB_ISNULL(word) || OB_UNLIKELY(0 >= word_len || 0 >= char_cnt || 0 >= word_freq)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (is_min_max_word(char_cnt)) {
    ++min_max_word_cnt_;
  } else if (OB_FAIL(casedown_word(src_word, dst_word))) {
  } else if (OB_FAIL(check_stopword(dst_word, is_stopword))) {
  } else if (OB_UNLIKELY(is_stopword)) {
    ++stopword_cnt_;
  } else if (OB_FAIL(groupby_word(dst_word, word_freq))) {
  } else {
    non_stopword_cnt_ += word_freq;
  }
  return ret;
}

bool ObAddWord::is_min_max_word(const int64_t c_len) const
{
  return flag_.min_max_word() && (c_len < min_token_size_ || c_len > max_token_size_);
}

int ObAddWord::casedown_word(const ObFTWord &src, ObFTWord &dst)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(src.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (flag_.casedown()) {
    ObString dst_str;
    if (OB_FAIL(ObCharset::tolower(
                    word_meta_.get_collation_type(),
                    src.get_word().get_string(),
                    dst_str,
                    allocator_))) {
    } else {
      ObFTWord tmp(dst_str.length(), dst_str.ptr(), word_meta_);
      dst = tmp;
    }
  } else {
    dst = src;
  }
  return ret;
}

int ObAddWord::check_stopword(const ObFTWord &ft_word, bool &is_stopword)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(ft_word.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (flag_.stopword()) {
    ObStopWordChecker *stop_word_checker = ObFTParseData::instance().stop_word_checker();
    if (OB_ISNULL(stop_word_checker)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(stop_word_checker->check_stopword(ft_word, is_stopword))) {
    }
  }
  return ret;
}

int ObAddWord::groupby_word(const ObFTWord &word, const int64_t word_freq)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(word.empty() || word_freq <= 0)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!flag_.groupby_word()) {
    if (OB_FAIL(word_map_.set_refactored(word, 1/*word count*/))) {
    }
  } else {
    auto update_word = [&word, word_freq](common::hash::HashMapPair<ObFTWord, int64_t> &entry) {
      // Preserve overwrite's latest representative key without replacing its node.
      entry.first = word;
      entry.second += word_freq;
    };
    if (OB_FAIL(word_map_.set_or_update(word, 1/*first word count*/, update_word))) {
    }
  }
  return ret;
}

} // end namespace storage
} // end namespace oceanbase
