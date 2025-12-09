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

// Heuristic: decide when it's worth parallelizing.
inline bool use_parallel(
  std::int64_t nlines, 
  const lsresize::Plan1D& plan)
{
  const double L   = static_cast<double>(plan.out_total);
  const double nnz = plan.row_ptr.empty()
                   ? 0.0
                   : static_cast<double>(plan.row_ptr.back());
  const double wavg  = (L > 0.0) ? (nnz / L) : 0.0;
  const double flops = 2.0 * static_cast<double>(nlines) * L * wavg;

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

  // Determine number of threads:
  //  - default: hardware_concurrency()
  //  - override: LSRESIZE_NUM_THREADS (if > 0)
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 1;
  }
  if (const char* e = std::getenv("LSRESIZE_NUM_THREADS")) {
    if (int v = std::atoi(e); v > 0) {
      hw = static_cast<unsigned>(v);
    }
  }

  const std::int64_t nthreads = std::min<std::int64_t>(hw, nlines);
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