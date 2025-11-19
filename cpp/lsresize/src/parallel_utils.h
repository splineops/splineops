// splineops/cpp/lsresize/src/parallel_utils.h
#pragma once

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <algorithm>

#if defined(LSRESIZE_WITH_OPENMP)
  #include <omp.h>
#endif

#if defined(LSRESIZE_WITH_STDTHREAD)
  #include <thread>
#endif

#include "resize1d.h"  // for lsresize::Plan1D

namespace lsresize {

// Heuristic: decide when it's worth parallelizing.
inline bool use_parallel(std::int64_t nlines, const lsresize::Plan1D& plan)
{
  const double L   = static_cast<double>(plan.out_total);
  const double nnz = plan.row_ptr.empty()
                   ? 0.0
                   : static_cast<double>(plan.row_ptr.back());
  const double wavg  = (L > 0.0) ? (nnz / L) : 0.0;
  const double flops = 2.0 * static_cast<double>(nlines) * L * wavg;

  double thr = 1e6;  // default FLOPs threshold
  if (const char* env = std::getenv("LSRESIZE_OMP_THRESHOLD")) {
    if (double t = std::atof(env); t > 0.0) thr = t;
  }

  return (nlines > 64) || (flops > thr);
}

// Centralized scheduler:
//   - If LSRESIZE_WITH_OPENMP && use_parallel → OpenMP
//   - Else if LSRESIZE_WITH_STDTHREAD && use_parallel → std::thread
//   - Else → serial (worker(0, nlines))
template <typename Worker>
inline void run_parallel_or_serial(std::int64_t nlines,
                                   const lsresize::Plan1D& plan,
                                   Worker&& worker)
{
  if (nlines <= 0) return;

#if defined(LSRESIZE_WITH_OPENMP)
  if (use_parallel(nlines, plan)) {
    #pragma omp parallel
    {
      int nth = omp_get_num_threads();
      int tid = omp_get_thread_num();

      std::int64_t chunk = (nlines + nth - 1) / nth;
      std::int64_t start = tid * chunk;
      std::int64_t end   = std::min<std::int64_t>(nlines, start + chunk);

      if (start < end) {
        worker(start, end);  // one call per thread, many lines
      }
    }
    return;
  }
#endif

#if defined(LSRESIZE_WITH_STDTHREAD)
  if (use_parallel(nlines, plan)) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (const char* e = std::getenv("LSRESIZE_NUM_THREADS")) {
      int v = std::atoi(e);
      if (v > 0) hw = static_cast<unsigned>(v);
    }

    std::int64_t nthreads = std::min<std::int64_t>(hw, nlines);
    std::int64_t chunk    = (nlines + nthreads - 1) / nthreads;

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));

    for (std::int64_t t = 0; t < nthreads; ++t) {
      std::int64_t start = t * chunk;
      std::int64_t end   = std::min<std::int64_t>(nlines, start + chunk);
      if (start >= end) break;

      threads.emplace_back([start, end, &worker]() {
        worker(start, end);
      });
    }

    for (auto& th : threads) th.join();
    return;
  }
#endif

  // Serial fallback
  worker(0, nlines);
}

} // namespace lsresize
