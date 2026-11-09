// CPU byte-transport checks; not a HIP kernel, Recall or performance test.
#include "../knn_staging.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool v) { if (!v) throw std::runtime_error("kNN staging mismatch"); }
static uint32_t value(size_t i, bool packed) {
  if (i % 37 == 0) return UINT32_MAX;
  const uint32_t bits = uint32_t(i * UINT64_C(2654435761));
  return packed ? bits % UINT32_C(0x00ffffff) : bits;
}
static void transport(size_t rows, unsigned columns, size_t batch, bool packed) {
  constexpr size_t guard = 64;
  const size_t entries = rows * columns, width = packed ? 3 : sizeof(uint32_t);
  std::vector<unsigned char> full(entries * width + 2 * guard, 0xa5);
  unsigned char* host = full.data() + guard;
  for (size_t offset = 0; offset < rows; offset += batch) {
    const size_t count = std::min(batch, rows - offset), n = count * columns;
    std::vector<unsigned char> device(n * width + 2 * guard, 0x5a);
    unsigned char* local = device.data() + guard;
    // Encode an independent batch according to the existing low16/high8
    // storage contract, including high bytes and UINT32_MAX sentinels.
    for (size_t i = 0; i < n; ++i) {
      uint32_t id = value(offset * columns + i, packed);
      if (packed) {
        const uint16_t low = uint16_t(id);
        std::memcpy(local + i * 2, &low, sizeof(low));
        local[n * 2 + i] = uint8_t(id >> 16);
      } else std::memcpy(local + i * sizeof(id), &id, sizeof(id));
    }
    seekdb_hipvs::KnnCopySlice slices[2];
    unsigned length = 0;
    require(seekdb_hipvs::make_knn_copy_slices(rows, columns, offset, count, packed, slices, length));
    require(length == (packed ? 2u : 1u));
    for (unsigned i = 0; i < length; ++i) {
      require(slices[i].device_offset <= n * width &&
              slices[i].bytes <= n * width - slices[i].device_offset);
      require(slices[i].host_offset <= entries * width &&
              slices[i].bytes <= entries * width - slices[i].host_offset);
      std::memcpy(host + slices[i].host_offset, local + slices[i].device_offset, slices[i].bytes);
    }
    for (size_t i = 0; i < guard; ++i)
      require(device[i] == 0x5a && device[device.size() - guard + i] == 0x5a);
  }
  // Decode the complete matrix as prune_graph does, rather than undoing
  // the slice calculation. An interleaved-3-byte copy fails this comparison.
  for (size_t i = 0; i < entries; ++i) {
    uint32_t got;
    if (packed) {
      uint16_t low;
      std::memcpy(&low, host + i * 2, sizeof(low));
      got = uint32_t(low) | uint32_t(host[entries * 2 + i]) << 16;
      if (got == UINT32_C(0x00ffffff)) got = UINT32_MAX;
    } else std::memcpy(&got, host + i * sizeof(got), sizeof(got));
    require(got == value(i, packed));
  }
  for (size_t i = 0; i < guard; ++i)
    require(full[i] == 0xa5 && full[full.size() - guard + i] == 0xa5);
}

int main() {
  // Both formats, nonzero offsets, odd entry counts and partial last batches.
  for (bool packed : {false, true}) {
    transport(4099, 192, 1024, packed);
    transport(4099, 256, 1024, packed);
    transport(103, 7, 13, packed);
    transport(31, 3, 1, packed);
  }
  transport(250000, 192, 1024, true);
  transport(250000, 256, 1024, true);
  seekdb_hipvs::KnnCopySlice slices[2];
  unsigned length = 9;
  require(!seekdb_hipvs::make_knn_copy_slices(10, 192, 0, 0, true, slices, length) && length == 0);
  require(!seekdb_hipvs::make_knn_copy_slices(10, 192, 10, 1, true, slices, length));
  require(!seekdb_hipvs::make_knn_copy_slices(10, 192, 9, 2, true, slices, length));
  require(!seekdb_hipvs::make_knn_copy_slices(SIZE_MAX, 192, 0, 1, true, slices, length));
  require(!seekdb_hipvs::make_knn_copy_slices(SIZE_MAX / 2, 1, 0, 1, false, slices, length));
  std::puts("PASS: both ID layouts, exact 250K matrix, tail batches, sentinels, bounds and overflow");
}
