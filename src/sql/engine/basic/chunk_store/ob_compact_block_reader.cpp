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

#define USING_LOG_PREFIX SQL_ENG

#include "sql/engine/basic/chunk_store/ob_compact_block_reader.h"

namespace oceanbase
{
using namespace common;

namespace sql
{

int ObCompactBlockReader::CurRowInfo::init(const ChunkRowMeta *row_meta, const uint8_t offset_width, const char *buf)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(row_meta) || OB_ISNULL(buf)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (offset_width != BASE_OFFSET_SIZE && offset_width != EXTENDED_OFFSET_SIZE) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    buf_ = buf;
    row_size_ = reinterpret_cast<const uint32_t*>(buf);
    // the n-th var_column in offset array.
    cur_var_offset_pos_ = 0;
    cur_data_ptr_ = sizeof(ObDatum) * row_meta->col_cnt_ + sizeof(ObChunkDatumStore::StoredRow);
    var_column_cnt_ = row_meta->col_cnt_ - row_meta->fixed_cnt_;
    int64_t var_offset_size = offset_width * (var_column_cnt_ + 1);
    bitmap_size_ = sql::ObBitVector::memory_size(row_meta->col_cnt_);
    bit_vec_ = sql::to_bit_vector(buf + HEAD_SIZE);
    data_offset_ = HEAD_SIZE + bitmap_size_ + var_offset_size; // the start of fixed data buffer
    fix_offset_ = data_offset_;
    var_col_end_offset_ = *row_size_ - data_offset_;
  }
  return ret;
}

int ObCompactBlockReader::get_row(const ObChunkDatumStore::StoredRow *&sr)
{
  int ret = OB_SUCCESS;
  sr = nullptr;
  if (OB_ISNULL(cur_blk_) || OB_ISNULL(row_meta_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (!blk_has_next_row()) {
    ret = OB_ITER_END;
  } else if (cur_pos_in_blk_ > cur_blk_->raw_size_ - sizeof(ObTempBlockStore::Block)) {
    ret = OB_INDEX_OUT_OF_RANGE;
  } else {
    int64_t size = 0;
    sr = nullptr;
    ObChunkDatumStore::StoredRow *tmp_sr = nullptr;
    if (OB_FAIL(get_stored_row_size(size))) {
    } else if (OB_FAIL(alloc_stored_row(tmp_sr, size))) {
    } else if (OB_ISNULL(tmp_sr)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (cur_row_offset_width_ == BASE_OFFSET_SIZE && OB_FAIL(get_stored_row<uint16_t>(tmp_sr))){
    } else if (cur_row_offset_width_ == EXTENDED_OFFSET_SIZE && OB_FAIL(get_stored_row<uint32_t>(tmp_sr)))  {
    } else {
      sr = tmp_sr;
    }
  }
  return ret;
}



// This view is private to FTS prescan replay. It never masquerades as a
// contiguous StoredRow: its value pointers refer to the pinned compact block.
int ObCompactBlockReader::peek_fts_datums(ObDatum *datums, const int64_t count, int64_t &row_size)
{
  int ret = OB_SUCCESS;
  row_size = 0;
  if (nullptr == cur_blk_ || nullptr == row_meta_) {
    ret = OB_ERR_UNEXPECTED;
  } else if (!blk_has_next_row()) {
    ret = OB_ITER_END;
  } else if (nullptr == datums || count < 4 || count > 5 || row_meta_->col_cnt_ != count
      || row_meta_->extra_size_ != 0 || row_meta_->fixed_cnt_ != count - 2
      || row_meta_->var_data_off_ != (count - 2) * 8
      || row_meta_->column_length_.count() != count || row_meta_->column_offset_.count() != count) {
    ret = OB_NOT_SUPPORTED;
  } else {
    const int64_t bitmap_size = ObBitVector::memory_size(count);
    const int64_t data_offset = HEAD_SIZE + bitmap_size + 3 * BASE_OFFSET_SIZE;
    const int64_t payload_size = cur_blk_->raw_size_ - static_cast<int64_t>(sizeof(ObTempBlockStore::Block));
    if (cur_pos_in_blk_ < 0 || payload_size < HEAD_SIZE || cur_pos_in_blk_ > payload_size - HEAD_SIZE) {
      ret = OB_INDEX_OUT_OF_RANGE;
    } else {
      const char *row = cur_blk_->payload_ + cur_pos_in_blk_;
      uint32_t stored_size = 0;
      uint8_t width = 0;
      MEMCPY(&stored_size, row, sizeof(stored_size));
      MEMCPY(&width, row + sizeof(stored_size), sizeof(width));
      if (stored_size < HEAD_SIZE || stored_size > payload_size - cur_pos_in_blk_) {
        ret = OB_INDEX_OUT_OF_RANGE;
      } else if (width == EXTENDED_OFFSET_SIZE) {
        ret = OB_NOT_SUPPORTED;
      } else if (width != BASE_OFFSET_SIZE || bitmap_size != sizeof(uint64_t)
          || stored_size < data_offset + row_meta_->var_data_off_) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        uint64_t nulls = 0;
        uint16_t offsets[3];
        MEMCPY(&nulls, row + HEAD_SIZE, sizeof(nulls));
        MEMCPY(offsets, row + HEAD_SIZE + bitmap_size, sizeof(offsets));
        if ((nulls & ((1ULL << count) - 1)) != 0) {
          ret = OB_NOT_SUPPORTED;
        } else if (offsets[0] != row_meta_->var_data_off_ || offsets[0] > offsets[1]
            || offsets[1] > offsets[2] || offsets[2] != stored_size - data_offset) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          int64_t var_col = 0;
          for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
            const int64_t length = row_meta_->column_length_[i];
            const int64_t offset = row_meta_->column_offset_[i];
            if (length == 0 && var_col < 2) {
              datums[i].pack_ = offsets[var_col + 1] - offsets[var_col];
              datums[i].ptr_ = row + data_offset + offsets[var_col++];
            } else if (length == 8 && offset >= 0 && offset <= row_meta_->var_data_off_ - 8) {
              datums[i].pack_ = 8;
              datums[i].ptr_ = row + data_offset + offset;
            } else {
              ret = OB_NOT_SUPPORTED;
            }
          }
          if (OB_SUCC(ret) && var_col != 2) ret = OB_NOT_SUPPORTED;
          if (OB_SUCC(ret)) row_size = stored_size;
        }
      }
    }
  }
  return ret;
}

template <typename T>
int ObCompactBlockReader::get_stored_row(ObChunkDatumStore::StoredRow *&sr)
{
  int ret = OB_SUCCESS;

  if (OB_ISNULL(row_meta_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(row_info_.init(row_meta_, cur_row_offset_width_, &cur_blk_->payload_[cur_pos_in_blk_]))) {
  } else if (OB_ISNULL(row_info_.buf_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const T *offset_array = reinterpret_cast<const T *>(row_info_.buf_ + HEAD_SIZE + row_info_.bitmap_size_);
    for (int64_t i = 0; OB_SUCC(ret) && i < row_meta_->col_cnt_; i++) {
      ObDatum *cur_datum = reinterpret_cast<ObDatum *>(sr->payload_ + i * sizeof(ObDatum));
      uint32_t len = 0;
      char *tmp_ptr = reinterpret_cast<char *>(sr) + row_info_.cur_data_ptr_;
      if (row_info_.bit_vec_->at(i)) {
        cur_datum->set_null();
        if (row_meta_->column_length_[i] == 0) {
          row_info_.cur_var_offset_pos_ ++;
        }
      } else if (row_meta_->column_length_[i] == 0) {
        T offset = offset_array[row_info_.cur_var_offset_pos_];
        if (row_info_.cur_var_offset_pos_ < row_info_.var_column_cnt_) {
          len = offset_array[row_info_.cur_var_offset_pos_ +  1] - offset;
        } else {
          ret = OB_INDEX_OUT_OF_RANGE;
        }
        if (OB_SUCC(ret)) {
          // set datum->len_, use the pack_ to conver the NULL_FLAG
          cur_datum->pack_ = len;
          // set data
          MEMCPY(tmp_ptr, row_info_.buf_ + row_info_.fix_offset_ + offset, len);
          // set datum->ptr_
          cur_datum->ptr_ = tmp_ptr;
          row_info_.cur_var_offset_pos_ ++;
        }
      } else {
        int32_t offset = row_meta_->column_offset_[i];
        len = row_meta_->column_length_[i];
        cur_datum->pack_ = len;
        MEMCPY(tmp_ptr, row_info_.buf_ + row_info_.fix_offset_ + offset, len);
        cur_datum->ptr_ = tmp_ptr;
      }
      row_info_.cur_data_ptr_ += len;
    }
    if (OB_SUCC(ret)) {
      sr->cnt_ = row_meta_->col_cnt_;
      sr->row_size_ =  row_info_.cur_data_ptr_;
      cur_pos_in_blk_ += *row_info_.row_size_;
      cur_row_in_blk_++;
    }
  }
  return ret;
}

int ObCompactBlockReader::alloc_stored_row(ObChunkDatumStore::StoredRow *&sr, const int64_t size)
{
  int ret = OB_SUCCESS;
  if (size <= sr_size_ ) {
  } else {
    if (OB_NOT_NULL(sr_buffer_) && OB_NOT_NULL(store_)) {
      store_->free(sr_buffer_, sr_size_);
      sr_buffer_ = nullptr;
    }
    sr_buffer_ = static_cast<char*>(store_->alloc(size));
    if (OB_ISNULL(sr_buffer_)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      sr_size_ = size;
    }
  }
  if (OB_SUCC(ret)) {
    MEMSET(sr_buffer_, 0, sr_size_);
    sr = reinterpret_cast<ObChunkDatumStore::StoredRow*>(sr_buffer_);
  }
  return ret;
}

int ObCompactBlockReader::get_stored_row_size(int64_t &size)
{
  int ret = OB_SUCCESS;
  size = 0;
  if (OB_ISNULL(cur_blk_) || OB_ISNULL(row_meta_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ObCompactBlockReader::calc_stored_row_size(&cur_blk_->payload_[cur_pos_in_blk_], row_meta_, size))) {
  } else {
    cur_row_offset_width_ = *reinterpret_cast<const int8_t*>(cur_blk_->payload_ + cur_pos_in_blk_ + sizeof(uint32_t));
  }
  return ret;
}

int ObCompactBlockReader::calc_stored_row_size(const char *compact_row, const ChunkRowMeta *row_meta, int64_t &size)
{
  int ret = OB_SUCCESS;
  size = 0;
  if (OB_ISNULL(compact_row) || OB_ISNULL(row_meta)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    int64_t bit_map_size = sql::ObBitVector::memory_size(row_meta->col_cnt_);
    uint32_t row_size = *reinterpret_cast<const uint32_t*>(compact_row);
    int64_t offset_width = *reinterpret_cast<const int8_t*>(compact_row + sizeof(uint32_t));
    int64_t var_column_cnt = row_meta->col_cnt_ - row_meta->fixed_cnt_;
    if (offset_width != BASE_OFFSET_SIZE && offset_width != EXTENDED_OFFSET_SIZE) {
      ret = OB_ERR_UNEXPECTED;
    }
    if (OB_SUCC(ret)) {
      size = row_size - HEAD_SIZE - bit_map_size - offset_width * (var_column_cnt + 1)\
            + row_meta->col_cnt_ * sizeof(ObDatum) + sizeof(ObChunkDatumStore::StoredRow);

    }
  }
  return ret;
}

}
}
