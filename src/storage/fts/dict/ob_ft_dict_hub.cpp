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

#include "storage/fts/dict/ob_ft_dict_hub.h"

#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/utility/ob_macro_utils.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_range_dict.h"
namespace oceanbase
{
namespace storage
{
int ObFTDictHub::init()
{
  static constexpr int K_MAX_DICT_BUCKET = 128; // for now, only built-in dicts.
  int ret = OB_SUCCESS;
  if (OB_FAIL(dict_map_.create(K_MAX_DICT_BUCKET, "dict_map"))) {
  } else if (OB_FAIL(rw_dict_lock_.init(K_MAX_DICT_BUCKET))) {
  } else {
    is_inited_ = true;
  }
  return ret;
};

int ObFTDictHub::destroy()
{
  int ret = OB_SUCCESS;
  is_inited_ = false;
  return ret;
}
int ObFTDictHub::build_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  ObFTDictInfo info;
  container.reset();

  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else {
    ObBucketHashWLockGuard guard(rw_dict_lock_, key.hash());

    // try if valid with no recursive lock
    if (OB_FAIL(get_dict_info(key, info))) {
      if (OB_HASH_NOT_EXIST == ret) {
        // dict not exist, make new one, by caller
        ret = OB_ENTRY_NOT_EXIST;
      } else {
      }
    } else if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
      } else {
      }
    }

    if (OB_FAIL(ret)) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        if (OB_FAIL(ObFTRangeDict::build_cache_from_ik_dict(desc, container))) {
        } else if (FALSE_IT(info.range_count_ = container.get_handles().size())) {
        } else if (OB_FAIL(put_dict_info(key, info))) {
        } else if (FALSE_IT(invalidate_built_dict(desc.type_))) {
        }
      }
    }
  }
  return ret;
}

class ObFTBuiltDictCache
{
public:
  ObFTBuiltDictCache(ObIAllocator &alloc) : container_(alloc), dict_(nullptr), version_(0) {}
public:
  ObFTCacheRangeContainer container_;
  ObIFTDict *dict_;
  int64_t version_;
};

static int built_dict_slot(const ObFTDictType type)
{
  int slot = -1;
  switch (type) {
  case ObFTDictType::DICT_IK_MAIN: slot = 0; break;
  case ObFTDictType::DICT_IK_QUAN: slot = 1; break;
  case ObFTDictType::DICT_IK_STOP: slot = 2; break;
  default: slot = -1; break;
  }
  return slot;
}

int ObFTDictHub::get_built_dict(const ObFTDictDesc &desc, ObIFTDict *&dict)
{
  int ret = OB_SUCCESS;
  dict = nullptr;
  const int slot = built_dict_slot(desc.type_);

  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else if (slot < 0) {
    ret = OB_NOT_SUPPORTED;
  } else {
    ObFTBuiltDictCache *cached = ATOMIC_LOAD(&built_dicts_[slot]);
    if (OB_NOT_NULL(cached)) {
      dict = cached->dict_;
    } else {
      // First use of this dictionary in the process: materialize the cache and
      // build the range dictionary once. Every later tokenize() call reuses it.
      ObFTBuiltDictCache *fresh = OB_NEWx(ObFTBuiltDictCache, &dict_cache_alloc_,
                                          dict_cache_alloc_);
      if (OB_ISNULL(fresh)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        ret = load_cache(desc, fresh->container_);
        if (OB_ENTRY_NOT_EXIST == ret) {
          ret = build_cache(desc, fresh->container_);
        }
        if (OB_SUCC(ret) && OB_ISNULL(fresh->dict_ =
                OB_NEWx(ObFTRangeDict, &dict_cache_alloc_, dict_cache_alloc_,
                        &fresh->container_, desc))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        }
        if (OB_SUCC(ret) && OB_FAIL(fresh->dict_->init())) {
        }
        if (OB_SUCC(ret)) {
          ObFTBuiltDictCache *expected = nullptr;
          if (!ATOMIC_CAS(&built_dicts_[slot], expected, fresh)) {
            // Another session won the race; both instances are identical and
            // read-only, so use the published one.
          }
          cached = ATOMIC_LOAD(&built_dicts_[slot]);
          dict = OB_NOT_NULL(cached) ? cached->dict_ : fresh->dict_;
        }
      }
    }
  }
  return ret;
}

void ObFTDictHub::invalidate_built_dict(const ObFTDictType type)
{
  const int slot = built_dict_slot(type);
  if (slot >= 0) {
    // Drop the shared instance so the next parser rebuilds it from the fresh
    // cache. The old instance is intentionally left to the hub allocator.
    ATOMIC_STORE(&built_dicts_[slot], nullptr);
  }
}

int ObFTDictHub::load_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfo info;
  container.reset();
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else {
    {
      ObBucketHashRLockGuard guard(rw_dict_lock_, key.hash());
      if (OB_FAIL(get_dict_info(key, info))) {
        if (OB_HASH_NOT_EXIST == ret) {
          // dict not exist, make new one, by caller
          ret = OB_ENTRY_NOT_EXIST;
        } else {
        }
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        // dict not exist, make new one, by caller
      } else {
      }
    }
  }

  return ret;
}


int ObFTDictHub::get_dict_info(const ObFTDictInfoKey &key, ObFTDictInfo &info)
{
  int ret = OB_SUCCESS;

  if (OB_FAIL(dict_map_.get_refactored(key, info))) {
    if (OB_HASH_NOT_EXIST != ret) {
    }
  }

  return ret;
}

int ObFTDictHub::put_dict_info(const ObFTDictInfoKey &key, const ObFTDictInfo &info)
{
  int ret = OB_SUCCESS;
  const int cover_exist_flag = 1;
  if (OB_FAIL(dict_map_.set_refactored(key, info, cover_exist_flag))) {
  }

  return ret;
}
} //  namespace storage
} //  namespace oceanbase
