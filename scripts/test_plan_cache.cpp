// Focused standalone regression for the native weighted Plan1D cache.
//
// Build from the repository root with the core resize sources and -pthread.

#include "resize_1d.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using CoeffSignVector =
    decltype(std::declval<lsresize::Plan1D>().coeff_sgn);
static_assert(std::is_same_v<CoeffSignVector, std::vector<std::int8_t>>);

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

const lsresize::LSParams kParams{3, 3, 3, 0.61, 0.0};

std::shared_ptr<const lsresize::Plan1D> plan(int size)
{
  return lsresize::get_plan_1d_cached(size, kParams);
}

void clear_cache()
{
  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "0");
  (void)plan(31);
}

bool cached_with_budget(int size, std::size_t budget)
{
  clear_cache();
  set_environment("LSRESIZE_PLAN_CACHE_SIZE", "32");
  const std::string text = std::to_string(budget);
  set_environment("LSRESIZE_PLAN_CACHE_BYTES", text.c_str());
  const auto first = plan(size);
  return plan(size).get() == first.get();
}

std::size_t minimum_cache_budget(int size)
{
  std::size_t upper = 1;
  while (!cached_with_budget(size, upper)) {
    upper *= 2;
    require(upper <= 16 * 1024 * 1024,
            "small test plan unexpectedly exceeded search bound");
  }
  std::size_t lower = 0;
  while (lower + 1 < upper) {
    const std::size_t middle = lower + (upper - lower) / 2;
    if (cached_with_budget(size, middle)) {
      upper = middle;
    } else {
      lower = middle;
    }
  }
  return upper;
}

void test_limits_and_lru()
{
  set_environment("LSRESIZE_PLAN_CACHE_SIZE", "32");

  clear_cache();
  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "1");
  const auto oversized_first = plan(1000);
  const auto oversized_second = plan(1000);
  require(oversized_first.get() != oversized_second.get(),
          "an entry larger than the byte budget must not be cached");

  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "invalid");
  const auto fallback_first = plan(1001);
  const auto fallback_second = plan(1001);
  require(fallback_first.get() == fallback_second.get(),
          "invalid byte limit must use the 128 MiB default");

  set_environment(
      "LSRESIZE_PLAN_CACHE_BYTES",
      "999999999999999999999999999999999999999999999999");
  const auto overflow_first = plan(1002);
  const auto overflow_second = plan(1002);
  require(overflow_first.get() == overflow_second.get(),
          "overflowing byte limit must use the default");

  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "  +1048576\t");
  const auto spaced_first = plan(1003);
  const auto spaced_second = plan(1003);
  require(spaced_first.get() == spaced_second.get(),
          "valid whitespace-delimited byte limit must be accepted");

  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "1048576");
  const auto retained = plan(1004);
  require(retained.get() == plan(1004).get(),
          "enabled cache must retain a repeated plan");
  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "0");
  const auto after_clear = plan(1004);
  require(retained.get() != after_clear.get(),
          "zero byte limit must clear existing entries");

  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "1048576");
  set_environment("LSRESIZE_PLAN_CACHE_SIZE", "1");
  const auto lru_first = plan(1005);
  (void)plan(1006);
  const auto lru_rebuilt = plan(1005);
  require(lru_first.get() != lru_rebuilt.get(),
          "entry count must remain an independent LRU limit");

  set_environment("LSRESIZE_PLAN_CACHE_SIZE", "0");
  const auto count_disabled_first = plan(1007);
  const auto count_disabled_second = plan(1007);
  require(count_disabled_first.get() != count_disabled_second.get(),
          "zero entry limit must disable and clear the cache");
}

void test_byte_weighted_lru_eviction()
{
  const std::size_t first_bytes = minimum_cache_budget(1100);
  const std::size_t second_bytes = minimum_cache_budget(1200);
  const std::size_t one_plan_budget = std::max(first_bytes, second_bytes);

  clear_cache();
  set_environment("LSRESIZE_PLAN_CACHE_SIZE", "32");
  const std::string text = std::to_string(one_plan_budget);
  set_environment("LSRESIZE_PLAN_CACHE_BYTES", text.c_str());
  const auto first = plan(1100);
  const auto second = plan(1200);
  require(plan(1200).get() == second.get(),
          "most-recent plan must survive byte eviction");
  require(plan(1100).get() != first.get(),
          "combined byte weight must evict the least-recent plan");
}

void test_double_checked_construction()
{
  clear_cache();
  set_environment("LSRESIZE_PLAN_CACHE_SIZE", "32");
  set_environment("LSRESIZE_PLAN_CACHE_BYTES", "134217728");

  constexpr int count = 8;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::shared_ptr<const lsresize::Plan1D>> results(count);
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (int index = 0; index < count; ++index) {
    threads.emplace_back([index, &ready, &start, &results]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      results[static_cast<std::size_t>(index)] = plan(12000);
    });
  }
  while (ready.load(std::memory_order_acquire) != count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (int index = 1; index < count; ++index) {
    require(results[0].get() == results[static_cast<std::size_t>(index)].get(),
            "racing builders must converge on one cached plan");
  }
}

#if !defined(_WIN32)

void test_cache_mutex_after_fork()
{
  (void)plan(128);
  std::atomic<bool> stop{false};
  std::thread mutator([&stop]() {
    int index = 0;
    while (!stop.load(std::memory_order_acquire)) {
      (void)plan(128 + index % 11);
      ++index;
    }
  });

  for (int iteration = 0; iteration < 20; ++iteration) {
    const pid_t child = ::fork();
    if (child == 0) {
      ::alarm(5);
      try {
        (void)plan(130);
      } catch (...) {
        ::_exit(51);
      }
      ::_exit(0);
    }
    if (child < 0) {
      stop.store(true, std::memory_order_release);
      mutator.join();
      throw std::runtime_error("fork failed");
    }
    int status = 0;
    if (::waitpid(child, &status, 0) != child ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      stop.store(true, std::memory_order_release);
      mutator.join();
      throw std::runtime_error("child cache access failed after fork");
    }
  }

  stop.store(true, std::memory_order_release);
  mutator.join();
}

#endif

} // namespace

int main()
{
  EnvironmentGuard size_guard("LSRESIZE_PLAN_CACHE_SIZE");
  EnvironmentGuard bytes_guard("LSRESIZE_PLAN_CACHE_BYTES");
  try {
    test_limits_and_lru();
    test_byte_weighted_lru_eviction();
    test_double_checked_construction();
#if !defined(_WIN32)
    test_cache_mutex_after_fork();
#endif
  } catch (const std::exception& exception) {
    std::cerr << "plan cache regression failed: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
