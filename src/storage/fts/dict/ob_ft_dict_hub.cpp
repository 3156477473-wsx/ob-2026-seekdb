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
#include "lib/alloc/ob_malloc_allocator.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/utility/ob_macro_utils.h"
#include "storage/fts/dict/ob_ft_cache_container.h"
#include "storage/fts/dict/ob_ft_dict_def.h"
#include "storage/fts/dict/ob_ft_range_dict.h"
#include "storage/fts/dict/ob_ft_dict.h"
namespace oceanbase
{
namespace storage
{
int ObFTDictHub::init()
{
  static constexpr int K_MAX_DICT_BUCKET = 128; // for now, only built-in dicts.
  int ret = OB_SUCCESS;
  if (OB_FAIL(dict_map_.create(K_MAX_DICT_BUCKET, "dict_map"))) {
  } else if (OB_FAIL(shared_dict_views_.create(K_MAX_DICT_BUCKET, "FTDictViews"))) {
  } else if (OB_FAIL(rw_dict_lock_.init(K_MAX_DICT_BUCKET))) {
  } else {
    is_inited_ = true;
  }
  return ret;
};

int ObFTDictHub::destroy()
{
  int ret = OB_SUCCESS;
  // The owner must have quiesced every parser borrower before this call.
  // A cold builder holds this mutex, never a bucket lock across construction.
  lib::ObMutexGuard guard(view_build_lock_);
  is_inited_ = false;
  auto *allocator = lib::ObMallocAllocator::get_instance();
  if (shared_dict_views_.created()) {
    for (auto it = shared_dict_views_.begin(); it != shared_dict_views_.end(); ++it) {
      ObIFTDict *view = it->second;
      OB_DELETEx(ObIFTDict, allocator, view);
    }
    shared_dict_views_.destroy();
  }
  if (dict_map_.created()) {
    dict_map_.destroy();
  }
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
        }
      }
    }
  }
  return ret;
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

int ObFTDictHub::init_single_dict(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container)
{
  int ret = OB_SUCCESS;
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  ObFTDictInfo info;
  bool need_build = false;
  {
    ObBucketHashRLockGuard guard(rw_dict_lock_, key.hash());
    if (OB_FAIL(get_dict_info(key, info))) {
      if (OB_HASH_NOT_EXIST == ret) {
        // The dict was never cached, it has to be built below.
        need_build = true;
        ret = OB_SUCCESS;
      } else {
      }
    }
  }
  if (OB_SUCC(ret) && !need_build) {
    if (OB_FAIL(ObFTRangeDict::try_load_cache(desc, info.range_count_, container))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        // The kv-cache entry is gone, rebuild it below.
        need_build = true;
        ret = OB_SUCCESS;
      } else {
      }
    }
  }
  if (OB_SUCC(ret) && need_build) {
    // build_cache takes the write lock by itself, so it must not be called
    // while the lock is held.
    if (OB_FAIL(build_cache(desc, container))) {
    }
  }
  return ret;
}

int ObFTDictHub::get_shared_view(const ObFTDictInfoKey &key, ObIFTDict *&dict_view)
{
  int ret = OB_SUCCESS;
  dict_view = nullptr;
  if (OB_FAIL(shared_dict_views_.get_refactored(key, dict_view))) {
    if (OB_HASH_NOT_EXIST == ret) {
      ret = OB_ENTRY_NOT_EXIST;
    } else {
    }
  }
  return ret;
}

int ObFTDictHub::set_shared_view(const ObFTDictInfoKey &key, ObIFTDict *dict_view)
{
  int ret = OB_SUCCESS;
  const int cover_exist_flag = 0;
  if (OB_FAIL(shared_dict_views_.set_refactored(key, dict_view, cover_exist_flag))) {
  }
  return ret;
}

int ObFTDictHub::build_dict_view(const ObFTDictDesc &desc, ObIFTDict *&dict_view)
{
  int ret = OB_SUCCESS;
  dict_view = nullptr;
  ObArenaAllocator container_alloc(lib::ObMemAttr("FTViewContainer"));
  ObFTCacheRangeContainer container(container_alloc);
  auto *allocator = lib::ObMallocAllocator::get_instance();
  if (OB_FAIL(init_single_dict(desc, container))) {
  } else {
    // Object storage is individually reclaimable; its range arena owns DATs.
    ObFTRangeDict *view = OB_NEWx(ObFTRangeDict, allocator, *allocator, nullptr, desc);
    if (OB_ISNULL(view)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      view->set_own_copies();
      if (OB_FAIL(view->init_with_container(desc, container))) {
        OB_DELETEx(ObFTRangeDict, allocator, view);
      } else {
        dict_view = view;
      }
    }
  }
  return ret;
}

int ObFTDictHub::borrow_dict_view(const ObFTDictDesc &desc, const ObIFTDict *&dict_view)
{
  int ret = OB_SUCCESS;
  dict_view = nullptr;
  ObFTDictInfoKey key(static_cast<uint64_t>(desc.type_));
  if (!is_inited_) {
    ret = OB_NOT_INIT;
  } else if (desc.charset_ != CHARSET_UTF8MB4 || desc.coll_type_ != CS_TYPE_UTF8MB4_BIN) {
    // The existing built-in cache key is a dict type, not an arbitrary charset.
    ret = OB_NOT_SUPPORTED;
  } else {
    {
      ObBucketHashRLockGuard guard(rw_dict_lock_, key.hash());
      ObIFTDict *cached = nullptr;
      ret = get_shared_view(key, cached);
      if (OB_ENTRY_NOT_EXIST == ret) {
        ret = OB_SUCCESS;
      } else if (OB_SUCC(ret)) {
        dict_view = cached;
      }
    }
    if (OB_SUCC(ret) && OB_ISNULL(dict_view)) {
      // Lock order: singleflight -> bucket. Never bucket -> singleflight.
      lib::ObMutexGuard build_guard(view_build_lock_);
      {
        ObBucketHashRLockGuard guard(rw_dict_lock_, key.hash());
        ObIFTDict *cached = nullptr;
        ret = get_shared_view(key, cached);
        if (OB_ENTRY_NOT_EXIST == ret) {
          ret = OB_SUCCESS;
        } else if (OB_SUCC(ret)) {
          dict_view = cached;
        }
      }
      // Recheck before building, so waiting first callers do not copy again.
      ObIFTDict *new_view = nullptr;
      if (OB_FAIL(ret) || OB_NOT_NULL(dict_view)) {
      } else if (OB_FAIL(build_dict_view(desc, new_view))) {
      } else if (OB_ISNULL(new_view)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        ObBucketHashWLockGuard guard(rw_dict_lock_, key.hash());
        ObIFTDict *cached = nullptr;
        ret = get_shared_view(key, cached);
        if (OB_SUCCESS == ret) {
          dict_view = cached;
        } else if (OB_ENTRY_NOT_EXIST == ret) {
          ret = set_shared_view(key, new_view);
          if (OB_SUCC(ret)) {
            dict_view = new_view;
            new_view = nullptr;
          }
        }
      }
      // Failed publication and any discarded candidate reclaim object + arena.
      auto *allocator = lib::ObMallocAllocator::get_instance();
      OB_DELETEx(ObIFTDict, allocator, new_view);
    }
  }
  return ret;
}

} //  namespace storage
} //  namespace oceanbase
