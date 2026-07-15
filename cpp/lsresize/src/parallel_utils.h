// splineops/cpp/lsresize/src/parallel_utils.h
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <thread>

#include "parallel_executor.h"
#include "resize_1d.h"  // for lsresize::Plan1D

namespace lsresize {

namespace detail {

inline bool is_ascii_space(char value) noexcept
{
  return value == ' ' || value == '\t' || value == '\n' ||
         value == '\r' || value == '\f' || value == '\v';
}

// Parse one strictly positive decimal integer, accepting surrounding ASCII
// whitespace and an optional leading '+'. Values above max_value saturate at
// that bound; malformed, negative, and zero values return zero.
inline std::int64_t parse_bounded_positive_integer(
  const char* text,
  std::int64_t max_value) noexcept
{
  if (text == nullptr || max_value <= 0) {
    return 0;
  }

  while (is_ascii_space(*text)) {
    ++text;
  }
  if (*text == '+') {
    ++text;
  }

  bool saw_digit = false;
  bool saturated = false;
  std::int64_t value = 0;
  for (; *text >= '0' && *text <= '9'; ++text) {
    saw_digit = true;
    const std::int64_t digit = static_cast<std::int64_t>(*text - '0');
    if (!saturated) {
      if (value > max_value / 10 ||
          (value == max_value / 10 && digit > max_value % 10)) {
        value = max_value;
        saturated = true;
      } else {
        value = value * 10 + digit;
      }
    }
  }

  while (is_ascii_space(*text)) {
    ++text;
  }
  if (!saw_digit || *text != '\0' || value <= 0) {
    return 0;
  }
  return value;
}

inline bool ascii_token_equals(const char* text, const char* expected) noexcept
{
  while (is_ascii_space(*text)) {
    ++text;
  }

  while (*expected != '\0') {
    char actual = *text;
    if (actual >= 'A' && actual <= 'Z') {
      actual = static_cast<char>(actual - 'A' + 'a');
    }
    if (actual != *expected) {
      return false;
    }
    ++text;
    ++expected;
  }

  while (is_ascii_space(*text)) {
    ++text;
  }
  return *text == '\0';
}

} // namespace detail

inline double estimate_axis_flops(
  std::int64_t nlines,
  const lsresize::Plan1D& plan)
{
  const double L   = static_cast<double>(plan.out_total);
  const double nnz = plan.row_ptr.empty()
                   ? 0.0
                   : static_cast<double>(plan.row_ptr.back());
  const double wavg  = (L > 0.0) ? (nnz / L) : 0.0;
  return 2.0 * static_cast<double>(nlines) * L * wavg;
}

inline std::int64_t explicit_thread_count(std::int64_t nlines)
{
  const std::int64_t useful_limit = std::min<std::int64_t>(
      nlines,
      static_cast<std::int64_t>(detail::kMaxParallelParticipants));
  return detail::parse_bounded_positive_integer(
      std::getenv("LSRESIZE_NUM_THREADS"), useful_limit);
}

// Heuristic: decide when it's worth parallelizing.
inline bool use_parallel(
  std::int64_t nlines,
  const lsresize::Plan1D& plan)
{
  if (nlines <= 1) {
    return false;
  }

  if (explicit_thread_count(nlines) > 1) {
    return true;
  }

  const double flops = estimate_axis_flops(nlines, plan);

  // Default FLOP threshold: ~1e6 operations
  double thr = 1e6;

  // Optional override via env:
  //   LSRESIZE_PARALLEL_THRESHOLD = minimum FLOPs to trigger multithreading
  if (const char* env = std::getenv("LSRESIZE_PARALLEL_THRESHOLD")) {
    if (double t = std::atof(env); t > 0.0) {
      thr = t;
    }
  }

  return (flops > thr) || (nlines > 64 && flops > 0.5 * thr);
}

inline unsigned hardware_threads()
{
  unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1U : hw;
}

inline std::int64_t round_up_thread_count(
  std::int64_t desired,
  std::int64_t max_threads)
{
  if (max_threads <= 1) {
    return max_threads;
  }

  desired = std::max<std::int64_t>(1, desired);
  std::int64_t threads = 1;
  while (threads < desired && threads < max_threads) {
    threads *= 2;
  }
  return std::min<std::int64_t>(threads, max_threads);
}

inline std::int64_t default_thread_count(
  std::int64_t nlines,
  const lsresize::Plan1D& plan,
  std::int64_t automatic_max_threads =
      static_cast<std::int64_t>(detail::kMaxParallelParticipants))
{
  const std::int64_t logical_threads =
      static_cast<std::int64_t>(hardware_threads());
  std::int64_t max_threads = std::min<std::int64_t>(
      {logical_threads,
       nlines,
       std::max<std::int64_t>(1, automatic_max_threads),
       static_cast<std::int64_t>(detail::kMaxParallelParticipants)});
  if (max_threads <= 1) {
    return max_threads;
  }

  const double flops = estimate_axis_flops(nlines, plan);

  // std::thread exposes logical CPUs only. On common SMT workstations, using
  // every logical CPU is often slower for these memory-heavy axis passes. Keep
  // larger default pools near a physical-core estimate; LSRESIZE_NUM_THREADS
  // remains the explicit escape hatch.
  if (logical_threads > 8) {
    const std::int64_t physical_core_estimate =
        std::max<std::int64_t>(1, (logical_threads + 1) / 2);
    max_threads = std::min<std::int64_t>(max_threads, physical_core_estimate);
  }

  // About 75k estimated multiply/add operations per worker keeps medium 2-D
  // cases on the default cap without forcing tiny passes to launch many workers.
  const double work_per_thread = 7.5e4;
  const std::int64_t desired = static_cast<std::int64_t>(
      std::ceil(flops / work_per_thread));
  return round_up_thread_count(desired, max_threads);
}

inline std::int64_t thread_count(
  std::int64_t nlines,
  const lsresize::Plan1D& plan,
  std::int64_t automatic_max_threads =
      static_cast<std::int64_t>(detail::kMaxParallelParticipants))
{
  if (nlines <= 0) {
    return 0;
  }

  const std::int64_t explicit_threads = explicit_thread_count(nlines);
  if (explicit_threads > 0) {
    return explicit_threads;
  }
  return default_thread_count(nlines, plan, automatic_max_threads);
}

inline bool persistent_threads_enabled()
{
  const char* value = std::getenv("LSRESIZE_PERSISTENT_THREADS");
  if (value == nullptr) {
    return true;
  }
  return !detail::ascii_token_equals(value, "0") &&
         !detail::ascii_token_equals(value, "false") &&
         !detail::ascii_token_equals(value, "no") &&
         !detail::ascii_token_equals(value, "off");
}

// Centralized scheduler (std::thread only):
//   - If use_parallel(...) is true → launch a thread pool
//   - Else → run worker(0, nlines) in the current thread
//
// The Worker functor must have the signature:
//    void operator()(std::int64_t start, std::int64_t end);
// where [start, end) is a range of 1-D "lines" to process.
template <typename Worker>
inline void run_parallel_or_serial(
  std::int64_t nlines,
  const lsresize::Plan1D& plan,
  Worker&& worker,
  std::int64_t automatic_max_threads =
      static_cast<std::int64_t>(detail::kMaxParallelParticipants),
  bool force_automatic_parallel = false)
{
  if (nlines <= 0) {
    return;
  }

  const bool explicitly_serial = explicit_thread_count(nlines) == 1;
  if (!use_parallel(nlines, plan) &&
      !(force_automatic_parallel && !explicitly_serial)) {
    // Serial fallback
    worker(0, nlines);
    return;
  }

  std::int64_t nthreads = thread_count(nlines, plan, automatic_max_threads);
  if (force_automatic_parallel && explicit_thread_count(nlines) == 0) {
    nthreads = std::min<std::int64_t>(
        {nlines,
         std::max<std::int64_t>(1, automatic_max_threads),
         static_cast<std::int64_t>(hardware_threads()),
         static_cast<std::int64_t>(detail::kMaxParallelParticipants)});
  }
  if (nthreads <= 1) {
    worker(0, nlines);
    return;
  }

  const std::int64_t chunk    = (nlines + nthreads - 1) / nthreads;

  // Waiting for the same bounded executor from one of its workers can
  // deadlock. Nested regions retain deterministic line ranges but run in the
  // current worker.
  if (detail::in_parallel_worker()) {
    worker(0, nlines);
    return;
  }

  if (persistent_threads_enabled()) {
    using WorkerType = std::decay_t<Worker>;
    auto shared_worker =
        std::make_shared<WorkerType>(std::forward<Worker>(worker));
    std::vector<detail::ParallelTask> tasks;
    tasks.reserve(static_cast<size_t>(nthreads));

    for (std::int64_t t = 0; t < nthreads; ++t) {
      const std::int64_t start = t * chunk;
      const std::int64_t end =
          std::min<std::int64_t>(nlines, start + chunk);
      if (start >= end) {
        break;
      }
      tasks.emplace_back([start, end, shared_worker]() {
        (*shared_worker)(start, end);
      });
    }

    detail::run_parallel_tasks(std::move(tasks));
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(nthreads));
  std::vector<std::exception_ptr> exceptions(
      static_cast<size_t>(nthreads));

  try {
    for (std::int64_t t = 0; t < nthreads; ++t) {
      const std::int64_t start = t * chunk;
      const std::int64_t end =
          std::min<std::int64_t>(nlines, start + chunk);
      if (start >= end) {
        break;
      }

      threads.emplace_back([start, end, t, &worker, &exceptions]() {
        try {
          worker(start, end);
        } catch (...) {
          exceptions[static_cast<size_t>(t)] = std::current_exception();
        }
      });
    }
  } catch (...) {
    for (auto& th : threads) {
      if (th.joinable()) {
        th.join();
      }
    }
    throw;
  }

  for (auto& th : threads) {
    th.join();
  }

  for (const auto& exception : exceptions) {
    if (exception) {
      std::rethrow_exception(exception);
    }
  }
}

} // namespace lsresize
