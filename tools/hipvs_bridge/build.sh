#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}
HIP_ARCH=${SEEKDB_HIP_ARCH:-gfx1100}
OUTPUT=${1:-$SCRIPT_DIR/libseekdb_cuvs_bridge.so}

# The offline build host need not expose the competition GPU. Do not let HIPCC
# infer a different device target (or platform) from the build host.
export HIP_PLATFORM=amd
ARCH_FLAGS=("--offload-arch=$HIP_ARCH")

[[ -x "$HIPCC" ]] || { echo "hipcc is not executable: $HIPCC" >&2; exit 2; }
[[ -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include" >&2
  exit 2
}
[[ -f "$HIPVS_ROOT/lib/libcuvs_c.so" ]] || {
  echo "libcuvs_c.so is missing under $HIPVS_ROOT/lib" >&2
  exit 2
}
for header in cuvs/neighbors/cagra.h cuvs/neighbors/ivf_pq.h; do
  [[ -f "$HIPVS_ROOT/include/$header" ]] || {
    echo "public hipVS C header is missing: $HIPVS_ROOT/include/$header" >&2
    exit 2
  }
done

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT_DIR=$(cd "$(dirname "$OUTPUT")" && pwd)
OUTPUT="$OUTPUT_DIR/$(basename "$OUTPUT")"
BUILD_DIR=$(mktemp -d "$OUTPUT_DIR/.seekdb-cuvs-bridge.XXXXXX")
cleanup() {
  rm -f -- "$BUILD_DIR/bridge.o" "$BUILD_DIR/graph.o" "$BUILD_DIR/libseekdb_cuvs_bridge.so"
  rmdir -- "$BUILD_DIR" 2>/dev/null || true
}
trap cleanup EXIT
export TMPDIR="$BUILD_DIR"
FLAGS=("${ARCH_FLAGS[@]}" -fPIC -O3 -std=c++17 -x hip -D__HIP_PLATFORM_AMD__
       -I"$HIPVS_ROOT/include" -I"$ROCM_ROOT/include")

"$HIPCC" "${FLAGS[@]}" \
  -c "$SCRIPT_DIR/seekdb_cuvs_bridge.c" -o "$BUILD_DIR/bridge.o"
"$HIPCC" "${FLAGS[@]}" \
  -DSEEKDB_IVFPQ_PROBES=32 -DSEEKDB_IVFPQ_BATCH=1024 -DSEEKDB_IVFPQ_REFINEMENT=4 \
  -DSEEKDB_IVFPQ_INTERNAL_BATCH=192 -DSEEKDB_IVFPQ_BATCH_EVENTS=0 -DSEEKDB_IVFPQ_MEMORY_INFO=1 \
  -c "$SCRIPT_DIR/ivfpq_graph.hip" -o "$BUILD_DIR/graph.o"

"$HIPCC" \
  "${ARCH_FLAGS[@]}" \
  -shared \
  "$BUILD_DIR/bridge.o" "$BUILD_DIR/graph.o" \
  -L"$HIPVS_ROOT/lib" \
  -Wl,-soname,libseekdb_cuvs_bridge.so \
  -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
  -lcuvs_c \
  -o "$BUILD_DIR/libseekdb_cuvs_bridge.so"
mv -- "$BUILD_DIR/libseekdb_cuvs_bridge.so" "$OUTPUT"

echo "built $OUTPUT (HIP platform=amd, target=$HIP_ARCH)"
