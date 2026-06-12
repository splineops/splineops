// splineops/cpp/lsresize/src/parallel_utils.h
#pragma once

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <thread>

#include "resize_1d.h"  // for lsresize::Plan1D

namespace lsresize {

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

// Heuristic: decide when it's worth parallelizing.
inline bool use_parallel(
  std::int64_t nlines,
  const lsresize::Plan1D& plan)
{
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

  return (nlines > 64) || (flops > thr);
}

inline unsigned hardware_threads()
{
  unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1U : hw;
}

inline std::int64_t explicit_thread_count(std::int64_t nlines)
{
  if (const char* e = std::getenv("LSRESIZE_NUM_THREADS")) {
    if (int v = std::atoi(e); v > 0) {
      return std::min<std::int64_t>(static_cast<std::int64_t>(v), nlines);
    }
  }
  return 0;
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
  const lsresize::Plan1D& plan)
{
  const std::int64_t logical_threads =
      static_cast<std::int64_t>(hardware_threads());
  std::int64_t max_threads = std::min<std::int64_t>(logical_threads, nlines);
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
  const lsresize::Plan1D& plan)
{
  if (nlines <= 0) {
    return 0;
  }

  const std::int64_t explicit_threads = explicit_thread_count(nlines);
  if (explicit_threads > 0) {
    return explicit_threads;
  }
  return default_thread_count(nlines, plan);
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
  Worker&& worker)
{
  if (nlines <= 0) {
    return;
  }

  if (!use_parallel(nlines, plan)) {
    // Serial fallback
    worker(0, nlines);
    return;
  }

  const std::int64_t nthreads = thread_count(nlines, plan);
  if (nthreads <= 1) {
    worker(0, nlines);
    return;
  }

  const std::int64_t chunk    = (nlines + nthreads - 1) / nthreads;

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(nthreads));

  for (std::int64_t t = 0; t < nthreads; ++t) {
    const std::int64_t start = t * chunk;
    const std::int64_t end   = std::min<std::int64_t>(nlines, start + chunk);
    if (start >= end) {
      break;
    }

    threads.emplace_back([start, end, &worker]() {
      worker(start, end);
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

} // namespace lsresize
