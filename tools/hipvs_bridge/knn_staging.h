/* Copyright (c) 2026. Licensed under the Apache License, Version 2.0. */
#pragma once
#include <cstddef>
#include <cstdint>

namespace seekdb_hipvs {
struct KnnCopySlice {
  size_t device_offset;
  size_t host_offset;
  size_t bytes;
};

// A packed batch is low16[batch_entries], high8[batch_entries]. Its two
// planes must land in the corresponding planes of the complete host graph;
// copying a batch as an interleaved 3-byte row would corrupt every later batch.
inline bool make_knn_copy_slices(size_t rows, unsigned columns, size_t offset,
                                 size_t count, bool packed,
                                 KnnCopySlice (&slices)[2], unsigned& length) {
  length = 0;
  if (!rows || !columns || !count || offset >= rows || count > rows - offset ||
      rows > SIZE_MAX / columns) return false;
  const size_t entries = rows * columns;
  const size_t width = packed ? 3 : sizeof(uint32_t);
  if (entries > SIZE_MAX / width) return false;
  const size_t start = offset * columns;
  const size_t batch_entries = count * columns;
  if (packed) {
    slices[0] = {0, start * 2, batch_entries * 2};
    slices[1] = {batch_entries * 2, entries * 2 + start, batch_entries};
    length = 2;
  } else {
    slices[0] = {0, start * sizeof(uint32_t), batch_entries * sizeof(uint32_t)};
    slices[1] = {0, 0, 0};
    length = 1;
  }
  return true;
}
} // namespace seekdb_hipvs
