# Host-only logic checks. The fake bridge is never linked into seekdb.
CC ?= gcc
CXX ?= g++
OUT ?= $(CURDIR)/build_host_tests
export TMPDIR := $(OUT)/tmp
HOST_FLAGS = -O1 -Wall -Wextra -Werror -UNDEBUG
FAKE_FLAGS = -D__HIP_PLATFORM_AMD__ -Istubs
FAKE_HEADERS = $(wildcard stubs/cuvs/*.h stubs/cuvs/core/*.h stubs/cuvs/neighbors/*.h stubs/dlpack/*.h)

.PHONY: all check
all: $(OUT)/test-query-worker $(OUT)/test-device-build-c $(OUT)/test-device-build-cpp

$(OUT)/tmp:
	mkdir -p "$(OUT)/tmp"

$(OUT)/test-query-worker: test_query_worker.cpp ../../../src/oblib/lib/vector/cuvs_query_worker.h | $(OUT)/tmp
	$(CXX) $(CXXFLAGS) -std=c++17 $(HOST_FLAGS) -pthread $< -o "$@"

$(OUT)/test-device-build-c: test_device_build_input.c ../seekdb_cuvs_bridge.c ../seekdb_cuvs_bridge.h ../ivfpq_graph.h $(FAKE_HEADERS) | $(OUT)/tmp
	$(CC) $(CFLAGS) -std=c11 $(HOST_FLAGS) $(FAKE_FLAGS) $< -o "$@"

$(OUT)/test-device-build-cpp: test_device_build_input.c ../seekdb_cuvs_bridge.c ../seekdb_cuvs_bridge.h ../ivfpq_graph.h $(FAKE_HEADERS) | $(OUT)/tmp
	$(CXX) $(CXXFLAGS) -x c++ -std=c++17 $(HOST_FLAGS) $(FAKE_FLAGS) $< -o "$@"

check: all
	timeout 20s "$(OUT)/test-query-worker"
	timeout 20s "$(OUT)/test-device-build-c"
	timeout 20s "$(OUT)/test-device-build-cpp"
