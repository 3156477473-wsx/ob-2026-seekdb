#pragma once

#include <exception>
#include <pthread.h>
#include <unistd.h>

namespace oceanbase {
namespace common {
namespace obvsag {

// A synchronous dispatcher, not a query batcher. The caller keeps every job
// argument alive until run() returns. Destruction excludes all callers.
class CuvsQueryWorker final {
public:
  CuvsQueryWorker() = default;
  CuvsQueryWorker(const CuvsQueryWorker &) = delete;
  CuvsQueryWorker &operator=(const CuvsQueryWorker &) = delete;

  ~CuvsQueryWorker()
  {
    // An inherited HIP context cannot be used after fork. Do not touch inherited
    // waiters or join a thread which exists only in the parent process.
    if (::getpid() != owner_pid_) { return; }
    if (current_worker() == this) { std::terminate(); }
    CancellationGuard cancellation_guard;
    {
      Lock submit_guard(submit_mu_);
      if (!submit_guard.locked()) { std::terminate(); }
      {
        Lock state_guard(state_mu_);
        if (!state_guard.locked()) { std::terminate(); }
        stopping_ = true;
      }
      if (started_) {
        pthread_cond_signal(&work_ready_);
        if (pthread_join(thread_, nullptr) != 0) { std::terminate(); }
      }
    }
    pthread_cond_destroy(&work_done_);
    pthread_cond_destroy(&work_ready_);
    pthread_mutex_destroy(&state_mu_);
    pthread_mutex_destroy(&submit_mu_);
  }

  bool run(void *(*callback)(void *), void *arg)
  {
    if (callback == nullptr || ::getpid() != owner_pid_ || current_worker() == this) {
      return false;
    }
    // A canceled SQL waiter must not unwind its stack while the GPU job still
    // uses that stack's pointers. Restore cancellation only after both locks go.
    CancellationGuard cancellation_guard;
    Lock submit_guard(submit_mu_);
    if (!submit_guard.locked()) { return false; }
    Lock state_guard(state_mu_);
    if (!state_guard.locked() || stopping_) { return false; }
    if (!started_) {
      pthread_attr_t attr;
      if (pthread_attr_init(&attr) != 0) { return false; }
      int rc = pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024);
      if (rc == 0) { rc = pthread_create(&thread_, &attr, worker_main, this); }
      pthread_attr_destroy(&attr);
      if (rc != 0) { return false; }
      started_ = true;
    }
    callback_ = callback;
    arg_ = arg;
    pending_ = true;
    pthread_cond_signal(&work_ready_);
    while (pending_) {
      if (pthread_cond_wait(&work_done_, &state_mu_) != 0) { std::terminate(); }
    }
    return true;
  }

private:
  class Lock {
  public:
    explicit Lock(pthread_mutex_t &mu) : mu_(mu), locked_(pthread_mutex_lock(&mu_) == 0) {}
    ~Lock() { if (locked_) { pthread_mutex_unlock(&mu_); } }
    bool locked() const { return locked_; }
  private:
    pthread_mutex_t &mu_;
    bool locked_;
  };
  class CancellationGuard {
  public:
    CancellationGuard()
    {
      if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_state_) != 0) { std::terminate(); }
    }
    ~CancellationGuard() noexcept(false) { pthread_setcancelstate(old_state_, nullptr); }
  private:
    int old_state_ = PTHREAD_CANCEL_ENABLE;
  };
  static CuvsQueryWorker *&current_worker()
  {
    static thread_local CuvsQueryWorker *worker = nullptr;
    return worker;
  }
  static void *worker_main(void *arg)
  {
    CuvsQueryWorker *worker = static_cast<CuvsQueryWorker *>(arg);
    current_worker() = worker;
    if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr) != 0 ||
        pthread_mutex_lock(&worker->state_mu_) != 0) { std::terminate(); }
    for (;;) {
      while (!worker->pending_ && !worker->stopping_) {
        if (pthread_cond_wait(&worker->work_ready_, &worker->state_mu_) != 0) { std::terminate(); }
      }
      if (worker->stopping_) { break; }
      void *(*callback)(void *) = worker->callback_;
      void *job = worker->arg_;
      pthread_mutex_unlock(&worker->state_mu_);
      callback(job);
      pthread_mutex_lock(&worker->state_mu_);
      worker->callback_ = nullptr;
      worker->arg_ = nullptr;
      worker->pending_ = false;
      pthread_cond_signal(&worker->work_done_);
    }
    pthread_mutex_unlock(&worker->state_mu_);
    current_worker() = nullptr;
    return nullptr;
  }

  const pid_t owner_pid_ = ::getpid();
  pthread_mutex_t submit_mu_ = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t state_mu_ = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t work_ready_ = PTHREAD_COND_INITIALIZER;
  pthread_cond_t work_done_ = PTHREAD_COND_INITIALIZER;
  pthread_t thread_{};
  void *(*callback_)(void *) = nullptr;
  void *arg_ = nullptr;
  bool started_ = false;
  bool stopping_ = false;
  bool pending_ = false;
};

}  // namespace obvsag
}  // namespace common
}  // namespace oceanbase
