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

#ifndef _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_HUB_H_
#define _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_HUB_H_

#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "lib/hash/ob_hashmap.h"
#include "lib/lock/ob_bucket_lock.h"
#include "lib/lock/ob_mutex.h"
#include "storage/fts/dict/ob_ft_dict_def.h"

namespace oceanbase
{
namespace storage
{
// typedef uint64_t ObFTTableID;
// typedef ObFTTableID ObFTDictTableID;

class ObFTDictInfo
{
public:
  ObFTDictInfo()
      : name_(""),
        type_(ObFTDictType::DICT_TYPE_INVALID),
        charset_(CHARSET_INVALID),
        version_(0),
        range_count_(0)
  {
  }

public:
  char name_[2048]; // for now
  ObFTDictType type_;
  ObCharsetType charset_;
  int64_t version_; // in memory
  int32_t range_count_;
};

struct ObFTDictInfoKey
{
public:
  ObFTDictInfoKey()
      : type_(static_cast<uint64_t>(ObFTDictType::DICT_TYPE_INVALID))
  {
  } // default constructor
  ObFTDictInfoKey(const uint64_t type)
      : type_(type)
  {
  }
  int hash(uint64_t &hash_value) const
  {
    int ret = OB_SUCCESS;
    hash_value = hash();
    return ret;
  }

  uint64_t hash() const
  {
    uint64_t hash = 0;
    hash = common::murmurhash(&type_, sizeof(int64_t), hash);
    return hash;
  }

  bool operator==(const ObFTDictInfoKey &other) const
  {
    return type_ == other.type_;
  }

  int compare(const ObFTDictInfoKey &other) const
  {
    int ret = 0;
    if (0 == ret) {
      ret = type_ - other.type_;
    }
    return ret;
  }

private:
  uint64_t type_;
  // name
};

class ObFTCacheRangeContainer;
class ObIFTDict;
class ObFTDictHub
{
public:
  ObFTDictHub()
      : is_inited_(false), dict_map_(), rw_dict_lock_(), shared_dict_views_()
  {
  }
  ~ObFTDictHub() { destroy(); }

  int init();

  // Lifecycle precondition: the owner has stopped/joined all parser users.
  // This is not concurrent borrower revocation; returned pointers are borrowed.
  int destroy();

  int build_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  int load_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  /**
   * @brief borrow a shared, read-only dict view.
   *
   * The view is built once and then reused by every parser, which avoids
   * rebuilding the range dict for every document. The view owns a private
   * copy of the dictionary blocks, so it does not depend on the kv-cache
   * handles of the thread that built it and stays valid for the whole
   * lifetime of the hub.
   *
   * The returned view is owned by the hub. Callers must not free it.
   */
  int borrow_dict_view(const ObFTDictDesc &desc, const ObIFTDict *&dict_view);

private:
  int get_dict_info(const ObFTDictInfoKey &key, ObFTDictInfo &info);

  int put_dict_info(const ObFTDictInfoKey &key, const ObFTDictInfo &info);

  // Load one dict into the given container, building the kv-cache entry when
  // it does not exist yet.
  int init_single_dict(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  // Build the shared view of one dict type.
  int build_dict_view(const ObFTDictDesc &desc, ObIFTDict *&dict_view);

  int get_shared_view(const ObFTDictInfoKey &key, ObIFTDict *&dict_view);

  int set_shared_view(const ObFTDictInfoKey &key, ObIFTDict *dict_view);

private:
  bool is_inited_;
  // holds info of dict
  hash::ObHashMap<ObFTDictInfoKey, ObFTDictInfo> dict_map_;
  ObBucketLock rw_dict_lock_;
  // Shared read-only dict views, keyed by the dict type. The views are
  // immutable once built and are owned by this hub.
  hash::ObHashMap<ObFTDictInfoKey, ObIFTDict *> shared_dict_views_;
  // Global cold singleflight; no bucket lock may be held when acquiring it.
  //
  // Each view owns its arena; this lock also prevents duplicate cold work.
  lib::ObMutex view_build_lock_;
};

} //  namespace storage
} //  namespace oceanbase

#endif // _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_HUB_H_
