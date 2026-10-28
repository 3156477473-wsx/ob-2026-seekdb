// CPU-only lifetime/concurrency checks; not HIP, SQL, Recall or performance evidence.
#include <atomic>
#include <cassert>
#include <cerrno>
#include <pthread.h>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sys/wait.h>
#include <unistd.h>
#endif

static int create_attempts = 0;
static bool fail_next_create = false;
static size_t requested_stack = 0;
static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*start)(void *), void *arg)
{
  ++create_attempts;
  assert(pthread_attr_getstacksize(attr, &requested_stack) == 0);
  if (fail_next_create) { fail_next_create = false; return EAGAIN; }
  return pthread_create(thread, attr, start, arg);
}
#define pthread_create test_pthread_create
#include "../../../src/oblib/lib/vector/cuvs_query_worker.h"
#undef pthread_create

using oceanbase::common::obvsag::CuvsQueryWorker;
struct Job {
  int input;
  int output = -1;
  pthread_t worker{};
};
static std::atomic<int> calls{0};
static std::atomic<int> active{0};
static std::atomic<int> live_worker_tls{0};
struct Lifetime {
  Lifetime() { ++live_worker_tls; }
  ~Lifetime() { --live_worker_tls; }
};
static void *execute(void *arg)
{
  static thread_local Lifetime lifetime;
  Job &job = *static_cast<Job *>(arg);
  assert(active.fetch_add(1) == 0);
  job.worker = pthread_self();
  std::this_thread::yield();
  job.output = job.input * 7;
  ++calls;
  assert(active.fetch_sub(1) == 1);
  return nullptr;
}
struct RecursiveJob { CuvsQueryWorker *worker; bool rejected = false; };
static void *recurse(void *arg)
{
  auto &job = *static_cast<RecursiveJob *>(arg);
  Job nested{9};
  job.rejected = !job.worker->run(execute, &nested);
  assert(nested.output == -1);
  return nullptr;
}
#ifdef __linux__
static std::atomic<bool> cancel_job_started{false};
static std::atomic<bool> cancel_job_release{false};
static std::atomic<bool> cancel_job_finished{false};
static void *blocking_execute(void *arg)
{
  cancel_job_started = true;
  while (!cancel_job_release.load()) { std::this_thread::yield(); }
  execute(arg);
  cancel_job_finished = true;
  return nullptr;
}
static void *cancel_waiter(void *arg)
{
  auto *worker = static_cast<CuvsQueryWorker *>(arg);
  Job job{7};  // This stack cannot disappear while blocking_execute still uses it.
  assert(worker->run(blocking_execute, &job));
  assert(job.output == 49);
  pthread_testcancel();
  return nullptr;
}
#endif

int main()
{
  { CuvsQueryWorker unused; }
  {
    CuvsQueryWorker worker;
    assert(!worker.run(nullptr, nullptr));
    Job first{2};
    fail_next_create = true;
    assert(!worker.run(execute, &first));
    assert(first.output == -1 && calls == 0);
    assert(worker.run(execute, &first));
    assert(first.output == 14);
    assert(requested_stack == 32UL * 1024 * 1024);
    const pthread_t original = first.worker;
    for (int i = 0; i < 4096; ++i) {
      Job job{i};
      assert(worker.run(execute, &job));
      assert(job.output == i * 7 && pthread_equal(original, job.worker));
    }
    std::vector<std::thread> callers;
    for (int t = 0; t < 4; ++t) {
      callers.emplace_back([&worker, original, t] {
        for (int i = 0; i < 128; ++i) {
          Job job{t * 128 + i};
          assert(worker.run(execute, &job));
          assert(job.output == job.input * 7 && pthread_equal(original, job.worker));
        }
      });
    }
    for (auto &caller : callers) { caller.join(); }
    assert(create_attempts == 2);  // one injected failure, one live worker
    assert(calls == 4609 && active == 0);
    assert(live_worker_tls == 1);
    RecursiveJob recursive{&worker};
    assert(worker.run(recurse, &recursive) && recursive.rejected);
  }  // waits for and joins the persistent worker
  assert(live_worker_tls == 0);
  {
    CuvsQueryWorker second;
    Job job{3};
    assert(second.run(execute, &job) && job.output == 21);
  }
  assert(create_attempts == 3 && calls == 4610 && active == 0);
  assert(live_worker_tls == 0);
#ifdef __linux__
  // A fork child rejects inherited state before locking or joining it. The
  // parent's worker still accepts work and releases its TLS after the child exits.
  auto *inherited = new CuvsQueryWorker();
  Job before_fork{4};
  assert(inherited->run(execute, &before_fork));
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    alarm(5);
    Job forbidden{5};
    if (inherited->run(execute, &forbidden)) { _exit(1); }
    delete inherited;
    _exit(0);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  Job after_fork{6};
  assert(inherited->run(execute, &after_fork) && after_fork.output == 42);
  delete inherited;
  assert(live_worker_tls == 0);
  {
    CuvsQueryWorker worker;
    pthread_t waiter;
    assert(pthread_create(&waiter, nullptr, cancel_waiter, &worker) == 0);
    while (!cancel_job_started.load()) { std::this_thread::yield(); }
    assert(pthread_cancel(waiter) == 0);
    cancel_job_release = true;
    void *result = nullptr;
    assert(pthread_join(waiter, &result) == 0 && result == PTHREAD_CANCELED);
    assert(cancel_job_finished.load());
    Job survives{8};
    assert(worker.run(execute, &survives) && survives.output == 56);
  }
  assert(live_worker_tls == 0);
#endif
}
