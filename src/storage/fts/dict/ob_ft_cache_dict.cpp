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

#include <cstdint>
#define USING_LOG_PREFIX STORAGE_FTS

#include "storage/fts/dict/ob_ft_cache_dict.h"

#include "lib/allocator/ob_allocator.h"
#include "lib/charset/ob_charset.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include "lib/utility/ob_macro_utils.h"
#include "share/rc/ob_server_runtime.h"
#include "storage/fts/dict/ob_ft_cache.h"
#include "storage/fts/dict/ob_ft_dict_def.h"

namespace oceanbase
{
namespace storage
{
ObFTCacheDict::~ObFTCacheDict()
{
  // Copies belong to the enclosing range view arena, not to the hub.
  // Destroying that view reclaims copies from successful and failed builds.
  owned_dat_ = nullptr;
}

int ObFTCacheDict::init()
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(owned_alloc_)) {
    // The dict owns a private copy of the dat block, copy it now.
    if (OB_ISNULL(dat_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const int64_t copy_size = dat_->mem_block_size_;
      char *buff = nullptr;
      if (OB_UNLIKELY(copy_size < static_cast<int64_t>(sizeof(ObFTDAT)))) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_ISNULL(buff = static_cast<char *>(owned_alloc_->alloc(copy_size)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        MEMCPY(buff, dat_, copy_size);
        owned_dat_ = reinterpret_cast<ObFTDAT *>(buff);
        dat_ = owned_dat_;
        reader_.set_dat(owned_dat_);
        if (ObCollationType::CS_TYPE_UTF8MB4_BIN == coll_type_) {
          // Build once before the owning shared view is published. Allocation
          // failure only disables this optional accelerator; the copied DAT
          // and the original map/cache lookup remain usable.
          auto *codes = static_cast<ObFTWordCode *>(
              owned_alloc_->alloc(ObArrayHashMap::BMP_CODE_COUNT * sizeof(ObFTWordCode)));
          if (nullptr != codes) {
            if (owned_dat_->get_map()->build_bmp_code_map(codes)) {
              reader_.set_dat(owned_dat_, codes);
            } else {
              owned_alloc_->free(codes);
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObFTCacheDict::match(const ObString &single_word, ObDATrieHit &hit,
                            ObFTCharCodeCache *char_cache) const
{
  return reader_.match_with_hit(single_word, hit, hit, char_cache);
}

int ObFTCacheDict::match(const ObString &words, bool &is_match) const
{
  int ret = OB_SUCCESS;

  int64_t char_len = 0;
  ObDATrieHit hit(this, 0);
  for (int64_t offset = 0; OB_SUCC(ret) && offset < words.length(); offset += char_len) {
    if (OB_FAIL(ObCharset::first_valid_char(coll_type_,
                                            words.ptr() + offset,
                                            words.length() - offset,
                                            char_len))) {
    } else if (OB_FAIL(match_with_hit(ObString(char_len, words.ptr() + offset), hit, hit))) {
    } else if (hit.is_match() && offset + char_len == words.length()) {
      is_match = true;
      break;
    } else if (hit.is_unmatch()) {
      is_match = false;
      break;
    } else {
      // continue to match
    }
  }
  return ret;
}

int ObFTCacheDict::match_with_hit(const ObString &single_word,
                                  const ObDATrieHit &last_hit,
                                  ObDATrieHit &hit,
                                  ObFTCharCodeCache *char_cache) const
{
  return reader_.match_with_hit(single_word, last_hit, hit, char_cache);
}

int ObFTCacheDict::make_and_fetch_cache_entry(const ObFTDictDesc &desc,
                                              ObFTDAT *dat_buff,
                                              const size_t buff_size,
                                              const int32_t range_id,
                                              const ObDictCacheValue *&value,
                                              ObKVCacheHandle &handle)
{
  int ret = OB_SUCCESS;
  ObDictCache &cache = ObDictCache::get_instance();
  uint64_t name = static_cast<uint64_t>(desc.type_);
  const ObDictCacheKey put_key(name, desc.type_, range_id);
  const ObDictCacheValue put_value(dat_buff);
  if (OB_FAIL(cache.put_and_fetch_dict(put_key, put_value, value, handle))) {
  }
  return ret;
}

} //  namespace storage
} //  namespace oceanbase
