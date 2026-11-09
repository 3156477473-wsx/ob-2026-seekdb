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

#ifndef _OCEANBASE_STORAGE_FTS_DICT_OB_FT_RANGE_DICT_H_
#define _OCEANBASE_STORAGE_FTS_DICT_OB_FT_RANGE_DICT_H_

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/ob_allocator.h"
#include "lib/allocator/page_arena.h"
#include "lib/container/ob_vector.h"
#include "lib/utility/ob_macro_utils.h"
#include "share/rc/ob_server_runtime.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_dict_iterator.h"

namespace oceanbase
{
namespace storage
{
class ObFTRangeDict final : public ObIFTDict
{
public:
  ObFTRangeDict(ObIAllocator &alloc,
                ObFTCacheRangeContainer *range_container,
                const ObFTDictDesc &desc)
      : is_inited_(false), desc_(desc), range_alloc_(lib::ObMemAttr("Range Dict")),
        range_dicts_(&range_alloc_), range_container_(range_container)
  {
  }
  ~ObFTRangeDict() override { destroy(); }

  /**
   * @brief make the range dict own a private copy of the dat blocks.
   *
   * The kv-cache handles that pin the dat blocks belong to the thread that
   * acquired them, so a range dict that is shared between threads must not
   * keep using them. When own_copies is set, every range dict copies the dat
   * block into this view's range arena and the range dict stops depending on the
   * kv-cache handles of the container.
   */
  void set_own_copies() { own_copies_ = true; }

public:
  struct ObFTRange final
  {
    ObFTSingleWord start_;
    ObFTSingleWord end_;
    ObIFTDict *dict_; // a cache dict
  };

public:
  int init() override;
  /**
   * @brief build the range dict from a container that the caller owns.
   *
   * The range dict does not keep the container, so the caller may release the
   * kv-cache handles of the container right after this call returns. This is
   * used by the shared dict views, which own private copies of the
   * dictionary blocks.
   */
  int init_with_container(const ObFTDictDesc &desc, const ObFTCacheRangeContainer &range_container);
  int match(const ObString &single_word, ObDATrieHit &hit,
                    ObFTCharCodeCache *char_cache = nullptr) const override;
  int match(const ObString &words, bool &is_match) const override;
  int match_with_hit(const ObString &single_word,
                     const ObDATrieHit &last_hit,
                     ObDATrieHit &hit,
                             ObFTCharCodeCache *char_cache = nullptr) const override;

public:
  int build_dict_from_cache(const ObFTCacheRangeContainer &range_container);

  static int try_load_cache(const ObFTDictDesc &desc,
                            const uint32_t range_count,
                            ObFTCacheRangeContainer &range_container);
  static int build_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &range_container);

  static int build_cache_from_ik_dict(const ObFTDictDesc &desc, ObFTCacheRangeContainer &range_container);

private:
  // build cache
  static int build_ranges(const ObFTDictDesc &desc,
                          ObIFTDictIterator &iter,
                          ObFTCacheRangeContainer &range_container);

  // build one range's cache
  static int build_one_range(const ObFTDictDesc &desc,
                             const int32_t range_id,
                             ObIFTDictIterator &iter,
                             ObFTCacheRangeContainer &container,
                             bool &build_next_range);

  static int build_ranges_concurrently_thread_pool(const ObFTDictDesc &desc,
                                                   ObIFTDictIterator &iter,
                                                   ObFTCacheRangeContainer &range_container);

private:
  void destroy()
  {
    for (int64_t i = 0; i < range_dicts_.size(); i++) {
      range_dicts_[i].dict_->~ObIFTDict(); // destroy dict
    }
    ObVector<ObFTRange, ObArenaAllocator> empty_ranges(&range_alloc_);
    // Empty assignment clears storage pointers before the arena resets.
    (void)range_dicts_.assign(empty_ranges);
    range_alloc_.reset();
    is_inited_ = false;
  }

  int find_first_char_range(const ObString &single_word, ObIFTDict *&dict) const;

private:
  static constexpr int DEFAULT_KEY_PER_RANGE = 50000; // by estimated
  bool is_inited_;
  ObFTDictDesc desc_;
  ObArenaAllocator range_alloc_;
  ObVector<ObFTRange, ObArenaAllocator> range_dicts_;
  ObFTCacheRangeContainer *range_container_; // only used to read cache
  // When set, every range dict owns a private copy of its dat block, so the
  // range dict stays valid after the kv-cache handles are released.
  bool own_copies_ = false;

private:
  DISALLOW_COPY_AND_ASSIGN(ObFTRangeDict);
};

} //  namespace storage
} //  namespace oceanbase

#endif // _OCEANBASE_STORAGE_FTS_DICT_OB_FT_RANGE_DICT_H_
