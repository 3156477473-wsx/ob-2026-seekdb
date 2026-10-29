#ifndef OCEANBASE_LIB_UTILITY_CGROUP_CPU_QUOTA_H_
#define OCEANBASE_LIB_UTILITY_CGROUP_CPU_QUOTA_H_

#include <stdint.h>

namespace oceanbase {
namespace common {

// Conservative whole-thread limit from all visible CPU quota ancestors.
// Returns 0 if unlimited, unavailable or malformed; callers must not use a
// host-wide CPU count to increase concurrency when this limit is unknown.
int64_t get_cgroup_cpu_quota_threads(
    const char *proc_cgroup_path = "/proc/self/cgroup",
    const char *proc_mountinfo_path = "/proc/self/mountinfo");

} // namespace common
} // namespace oceanbase

#endif
