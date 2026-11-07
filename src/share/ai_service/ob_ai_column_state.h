/**
 * Copyright (c) 2021 OceanBase
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OCEANBASE_SHARE_AI_SERVICE_OB_AI_COLUMN_STATE_H_
#define OCEANBASE_SHARE_AI_SERVICE_OB_AI_COLUMN_STATE_H_

#include <mutex>
#include <string.h>

#include "lib/ob_define.h"

namespace oceanbase
{
namespace share
{

// Readiness of one AI column, at column granularity: a statement that
// references an AI column must be rejected while any row of that column is
// still waiting for the model (or failed to be computed at all).
class ObAIColumnState
{
public:
  enum State
  {
    READY = 0,
    PENDING = 1,
    FAILED = 2
  };
  static const int64_t MAX_ENTRIES = 256;
  static const int64_t MAX_NAME_LEN = 96;

  ObAIColumnState() : count_(0), victim_(0) {}
  ~ObAIColumnState() {}

  // Records the readiness of one AI column. The owner of the tasks decides the
  // new state, so the check below only has to read it back.
  void mark(const uint64_t table_id, const char *name, const int64_t name_len,
            const int state)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    Entry *entry = find_or_create(table_id, name, name_len);
    if (NULL != entry) {
      entry->state_ = state;
    }
  }

  // Columns that were never touched are considered ready.
  int get_state(const uint64_t table_id, const char *name, const int64_t name_len) const
  {
    int state = READY;
    std::lock_guard<std::mutex> guard(mutex_);
    const Entry *entry = find(table_id, name, name_len);
    if (NULL != entry) {
      state = entry->state_;
    }
    return state;
  }

private:
  struct Entry
  {
    uint64_t table_id_;
    int64_t name_len_;
    char name_[MAX_NAME_LEN];
    int state_;
    bool used_;
  };

  const Entry *find(const uint64_t table_id, const char *name, const int64_t name_len) const
  {
    const Entry *found = NULL;
    for (int64_t i = 0; NULL == found && i < count_; ++i) {
      if (entries_[i].used_ && entries_[i].table_id_ == table_id
          && entries_[i].name_len_ == name_len
          && 0 == strncmp(entries_[i].name_, name, static_cast<size_t>(name_len))) {
        found = &entries_[i];
      }
    }
    return found;
  }

  Entry *find_or_create(const uint64_t table_id, const char *name, const int64_t name_len)
  {
    Entry *entry = const_cast<Entry *>(find(table_id, name, name_len));
    if (NULL == entry && name_len > 0 && name_len < MAX_NAME_LEN) {
      int64_t slot = count_;
      if (slot >= MAX_ENTRIES) {
        slot = victim_++ % MAX_ENTRIES;
      } else {
        ++count_;
      }
      entry = &entries_[slot];
      entry->table_id_ = table_id;
      entry->name_len_ = name_len;
      MEMCPY(entry->name_, name, name_len);
      entry->name_[name_len] = '\0';
      entry->state_ = READY;
      entry->used_ = true;
    }
    return entry;
  }

  Entry entries_[MAX_ENTRIES];
  int64_t count_;
  int64_t victim_;
  mutable std::mutex mutex_;
};

inline ObAIColumnState &ai_column_state()
{
  static ObAIColumnState state;
  return state;
}

}  // namespace share
}  // namespace oceanbase

#endif  // OCEANBASE_SHARE_AI_SERVICE_OB_AI_COLUMN_STATE_H_
