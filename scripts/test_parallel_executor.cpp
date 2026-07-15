// Focused standalone regression for the persistent resize executor.
//
// Build from the repository root with:
//   c++ -std=c++17 -pthread -Icpp/lsresize/src cpp/lsresize/src/parallel_executor.cpp scripts/test_parallel_executor.cpp -o /tmp/test_parallel_executor
//
// This is deliberately independent of Python and the resize kernels so fork
// recovery and scheduler environment parsing can be exercised in isolation.

#include "parallel_executor.h"
#include "parallel_utils.h"

#include <atomic>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

void require(bool condition, const char* message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void set_environment(const char* name, const char* value)
{
#if defined(_WIN32)
  if (_putenv_s(name, value) != 0) {
    throw std::runtime_error("could not set test environment variable");
  }
#else
  if (::setenv(name, value, 1) != 0) {
    throw std::runtime_error("could not set test environment variable");
  }
#endif
}

void unset_environment(const char* name)
{
#if defined(_WIN32)
  if (_putenv_s(name, "") != 0) {
    throw std::runtime_error("could not unset test environment variable");
  }
#else
  if (::unsetenv(name) != 0) {
    throw std::runtime_error("could not unset test environment variable");
  }
#endif
}

class EnvironmentGuard {
public:
  explicit EnvironmentGuard(const char* name)
      : name_(name)
  {
    if (const char* value = std::getenv(name)) {
      previous_ = value;
    }
  }

  ~EnvironmentGuard()
  {
    try {
      if (previous_) {
        set_environment(name_.c_str(), previous_->c_str());
      } else {
        unset_environment(name_.c_str());
      }
    } catch (...) {
      std::terminate();
    }
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

void test_environment_parsing()
{
  EnvironmentGuard thread_guard("LSRESIZE_NUM_THREADS");
  EnvironmentGuard persistent_guard("LSRESIZE_PERSISTENT_THREADS");

  unset_environment("LSRESIZE_NUM_THREADS");
  require(lsresize::explicit_thread_count(1000) == 0,
          "unset thread count must select the automatic policy");

  set_environment("LSRESIZE_NUM_THREADS", "  +4\t");
  require(lsresize::explicit_thread_count(1000) == 4,
          "surrounding whitespace and leading plus must be accepted");

  set_environment("LSRESIZE_NUM_THREADS", "999999999999999999999999999");
  require(
      lsresize::explicit_thread_count(1000) ==
          static_cast<std::int64_t>(
              lsresize::detail::kMaxParallelParticipants),
      "overflowing thread count must saturate at the executor cap");

  set_environment("LSRESIZE_NUM_THREADS", "999999999999999999999999999");
  require(lsresize::explicit_thread_count(3) == 3,
          "thread count must not exceed useful line work");

  for (const char* invalid : {"", "0", "-1", "4workers", "+", "  "}) {
    set_environment("LSRESIZE_NUM_THREADS", invalid);
    require(lsresize::explicit_thread_count(1000) == 0,
            "malformed thread count must select the automatic policy");
  }

  unset_environment("LSRESIZE_PERSISTENT_THREADS");
  require(lsresize::persistent_threads_enabled(),
          "persistent executor must remain enabled by default");
  for (const char* disabled : {"0", "FALSE", " No ", "\toFf\n"}) {
    set_environment("LSRESIZE_PERSISTENT_THREADS", disabled);
    require(!lsresize::persistent_threads_enabled(),
            "documented false token must disable persistent workers");
  }
  for (const char* enabled : {"1", "true", "yes", "invalid"}) {
    set_environment("LSRESIZE_PERSISTENT_THREADS", enabled);
    require(lsresize::persistent_threads_enabled(),
            "all non-false tokens must preserve enabled behavior");
  }

  lsresize::Plan1D plan{};
  plan.out_total = 100;
  plan.row_ptr = {0, 1000};
  unset_environment("LSRESIZE_NUM_THREADS");
  require(lsresize::thread_count(1000, plan, 8) <= 8,
          "automatic thread ceilings must bound the default scheduler");
  set_environment("LSRESIZE_NUM_THREADS", "16");
  require(lsresize::thread_count(1000, plan, 8) == 16,
          "explicit thread counts must override automatic ceilings");
}

void test_nested_exception_runs_every_task()
{
  std::atomic<int> completed{0};
  std::vector<lsresize::detail::ParallelTask> outer;
  outer.emplace_back([]() {});
  outer.emplace_back([&completed]() {
    require(lsresize::detail::in_parallel_worker(),
            "second outer task must execute on a pool worker");

    std::vector<lsresize::detail::ParallelTask> nested;
    nested.emplace_back([]() { throw std::runtime_error("expected"); });
    nested.emplace_back([&completed]() { completed.fetch_add(1); });
    nested.emplace_back([&completed]() { completed.fetch_add(1); });
    lsresize::detail::run_parallel_tasks(std::move(nested));
  });

  bool threw = false;
  try {
    lsresize::detail::run_parallel_tasks(std::move(outer));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "nested task exception must reach the submitting thread");
  require(completed.load() == 2,
          "nested executor must finish later tasks before rethrowing");
}

#if !defined(_WIN32)

void test_fork_from_worker_resets_nested_marker()
{
  std::atomic<int> child_status{-1};
  std::vector<lsresize::detail::ParallelTask> tasks;
  tasks.emplace_back([]() {});
  tasks.emplace_back([&child_status]() {
    require(lsresize::detail::in_parallel_worker(),
            "fork task must execute on a pool worker");

    const pid_t child = ::fork();
    if (child == 0) {
      if (lsresize::detail::in_parallel_worker()) {
        ::_exit(41);
      }

      std::atomic<int> completed{0};
      std::vector<lsresize::detail::ParallelTask> child_tasks;
      for (int i = 0; i < 4; ++i) {
        child_tasks.emplace_back(
            [&completed]() { completed.fetch_add(1); });
      }
      try {
        lsresize::detail::run_parallel_tasks(std::move(child_tasks));
      } catch (...) {
        ::_exit(42);
      }
      ::_exit(completed.load() == 4 ? 0 : 43);
    }
    if (child < 0) {
      throw std::runtime_error("fork failed");
    }

    int status = 0;
    if (::waitpid(child, &status, 0) != child) {
      throw std::runtime_error("waitpid failed");
    }
    if (WIFEXITED(status)) {
      child_status.store(WEXITSTATUS(status));
    } else {
      child_status.store(44);
    }
  });

  lsresize::detail::run_parallel_tasks(std::move(tasks));
  require(child_status.load() == 0,
          "forked child must create and use a fresh persistent executor");
}

#endif

} // namespace

int main()
{
  try {
    test_environment_parsing();
    test_nested_exception_runs_every_task();
#if !defined(_WIN32)
    test_fork_from_worker_resets_nested_marker();
#endif
  } catch (const std::exception& exception) {
    std::cerr << "parallel executor regression failed: "
              << exception.what() << '\n';
    return 1;
  }
  return 0;
}
