// splineops/cpp/lsresize/src/parallel_executor.cpp
#include "parallel_executor.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace lsresize::detail {
namespace {

thread_local bool g_parallel_worker = false;
#if !defined(_WIN32)
thread_local pid_t g_parallel_worker_pid = 0;
#endif

class TaskBatch {
public:
  explicit TaskBatch(std::vector<ParallelTask> tasks)
      : tasks_(std::move(tasks)),
        remaining_(tasks_.size())
  {
  }

  TaskBatch(const TaskBatch&) = delete;
  TaskBatch& operator=(const TaskBatch&) = delete;

  std::size_t size() const noexcept
  {
    return tasks_.size();
  }

  void run(std::size_t index) noexcept
  {
    try {
      tasks_[index]();
    } catch (...) {
      std::lock_guard<std::mutex> lock(exception_mutex_);
      if (!exception_) {
        exception_ = std::current_exception();
      }
    }

    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // Serialize the zero transition with wait()'s unlock-and-sleep step.
      // Notifying an atomic predicate without taking this mutex can lose the
      // wake-up between the predicate check and the condition-variable wait.
      std::lock_guard<std::mutex> lock(completion_mutex_);
      completion_cv_.notify_one();
    }
  }

  void wait()
  {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this]() {
      return remaining_.load(std::memory_order_acquire) == 0;
    });
    lock.unlock();

    std::exception_ptr exception;
    {
      std::lock_guard<std::mutex> exception_lock(exception_mutex_);
      exception = exception_;
    }
    if (exception) {
      std::rethrow_exception(exception);
    }
  }

private:
  std::vector<ParallelTask> tasks_;
  std::atomic<std::size_t> remaining_;
  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  std::mutex exception_mutex_;
  std::exception_ptr exception_;
};

struct QueuedTask {
  std::shared_ptr<TaskBatch> batch;
  std::size_t index;
};

class PersistentExecutor {
public:
  PersistentExecutor() = default;

  ~PersistentExecutor()
  {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stopping_ = true;
    }
    queue_cv_.notify_all();

    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  PersistentExecutor(const PersistentExecutor&) = delete;
  PersistentExecutor& operator=(const PersistentExecutor&) = delete;

  void run(std::vector<ParallelTask> tasks)
  {
    if (tasks.empty()) {
      return;
    }

    // A nested invocation cannot safely wait for the same bounded executor.
    // Execute its already-static task ranges in the current worker instead.
    // Preserve the top-level contract by running every range before rethrowing
    // the first exception.
    if (in_parallel_worker()) {
      std::exception_ptr exception;
      for (ParallelTask& task : tasks) {
        try {
          task();
        } catch (...) {
          if (!exception) {
            exception = std::current_exception();
          }
        }
      }
      if (exception) {
        std::rethrow_exception(exception);
      }
      return;
    }

    const auto batch = std::make_shared<TaskBatch>(std::move(tasks));
    // The submitting thread owns one static chunk. This keeps the requested
    // compute concurrency unchanged while avoiding a sleeping caller and one
    // extra worker wake-up per axis pass.
    ensure_worker_count(batch->size() - 1);
    enqueue(batch, 1);
    batch->run(0);
    batch->wait();
  }

private:
  void ensure_worker_count(std::size_t requested)
  {
    // The caller executes one task, so at most N-1 persistent workers are
    // useful.  Keep this second-line cap even when callers bypass the standard
    // line scheduler and submit a task vector directly.
    requested = std::min(requested, kMaxParallelParticipants - 1);
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (workers_.size() < requested) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  void enqueue(
      const std::shared_ptr<TaskBatch>& batch,
      std::size_t first_index)
  {
    if (first_index >= batch->size()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);

      // Reclaim consumed prefix storage while preserving FIFO order among
      // batches submitted concurrently by independent callers.
      if (queue_head_ == queue_.size()) {
        queue_.clear();
        queue_head_ = 0;
      } else if (queue_head_ >= 1024 && queue_head_ * 2 >= queue_.size()) {
        queue_.erase(
            queue_.begin(),
            queue_.begin() + static_cast<std::ptrdiff_t>(queue_head_));
        queue_head_ = 0;
      }

      // Reserve before modifying the queue so allocation failure leaves the
      // submission all-or-nothing. QueuedTask has noexcept move operations.
      const std::size_t enqueue_count = batch->size() - first_index;
      queue_.reserve(queue_.size() + enqueue_count);
      for (std::size_t i = first_index; i < batch->size(); ++i) {
        queue_.push_back(QueuedTask{batch, i});
      }
    }
    queue_cv_.notify_all();
  }

  void worker_loop()
  {
    g_parallel_worker = true;
#if !defined(_WIN32)
    g_parallel_worker_pid = ::getpid();
#endif
    for (;;) {
      QueuedTask task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() {
          return stopping_ || queue_head_ < queue_.size();
        });

        if (stopping_ && queue_head_ == queue_.size()) {
          break;
        }

        task = std::move(queue_[queue_head_]);
        ++queue_head_;
      }
      task.batch->run(task.index);
    }
    g_parallel_worker = false;
  }

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::vector<QueuedTask> queue_;
  std::size_t queue_head_ = 0;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};

#if !defined(_WIN32)

// std::thread objects inherited across fork still appear joinable even though
// their worker threads no longer exist. Keep the process-local executor behind
// a holder that abandons (rather than destroys) the inherited object in the
// child and lazily creates a fresh pool there.
class ExecutorHolder;

ExecutorHolder*& active_holder() noexcept
{
  static ExecutorHolder* holder = nullptr;
  return holder;
}

class ExecutorHolder {
public:
  ExecutorHolder()
      : pid_(::getpid())
  {
    active_holder() = this;
    (void)::pthread_atfork(
        &ExecutorHolder::prepare_fork,
        &ExecutorHolder::parent_after_fork,
        &ExecutorHolder::child_after_fork);
  }

  ~ExecutorHolder()
  {
    active_holder() = nullptr;
  }

  PersistentExecutor& get()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const pid_t current_pid = ::getpid();
    if (current_pid != pid_) {
      // Fallback for platforms or embedding situations where atfork handler
      // registration failed. Never destroy inherited joinable std::threads.
      (void)executor_.release();
      pid_ = current_pid;
      g_parallel_worker = false;
      g_parallel_worker_pid = current_pid;
    }
    if (!executor_) {
      executor_ = std::make_unique<PersistentExecutor>();
    }
    return *executor_;
  }

private:
  static void prepare_fork() noexcept
  {
    if (ExecutorHolder* holder = active_holder()) {
      holder->mutex_.lock();
    }
  }

  static void parent_after_fork() noexcept
  {
    if (ExecutorHolder* holder = active_holder()) {
      holder->mutex_.unlock();
    }
  }

  static void child_after_fork() noexcept
  {
    // A fork initiated by a pool worker copies that worker's thread-local
    // marker into the child.  It must not make every later parallel region in
    // the single-threaded child look nested.
    g_parallel_worker = false;
    g_parallel_worker_pid = ::getpid();

    if (ExecutorHolder* holder = active_holder()) {
      // The copied pool cannot be joined or reused in the child. Its memory is
      // intentionally abandoned in this process; the OS reclaimed the actual
      // parent threads at fork.
      (void)holder->executor_.release();
      holder->pid_ = ::getpid();
      holder->mutex_.unlock();
    }
  }

  std::mutex mutex_;
  std::unique_ptr<PersistentExecutor> executor_;
  pid_t pid_;
};

PersistentExecutor& executor()
{
  static ExecutorHolder holder;
  return holder.get();
}

#else

PersistentExecutor& executor()
{
  static PersistentExecutor instance;
  return instance;
}

#endif

} // namespace

void run_parallel_tasks(std::vector<ParallelTask> tasks)
{
  executor().run(std::move(tasks));
}

bool in_parallel_worker() noexcept
{
#if !defined(_WIN32)
  // This also covers the rare case where pthread_atfork registration failed:
  // only the calling thread survives and its copied marker belongs to the
  // parent process, not to a live worker in this child.
  if (g_parallel_worker && g_parallel_worker_pid != ::getpid()) {
    g_parallel_worker = false;
    g_parallel_worker_pid = ::getpid();
  }
#endif
  return g_parallel_worker;
}

} // namespace lsresize::detail
