// splineops/cpp/lsresize/src/resize_1d.cpp
#include "resize_1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"
#include "dot_kernels.h"
#include "profile_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

// 1D resizing pipeline layout
//
//   Public entry points (see resize_1d.h):
//
//     - resize_1d_workspace(in_vec, out_vec, params, plan, workspace)
//     - resize_1d_line_contiguous(in_ptr, out_ptr, params, plan, workspace)
//     - resize_1d_line_buffered(line_buf, out_vec, params, plan, workspace)
//
// All three delegate to a single internal pipeline:
//
//   run_pipeline_from_line(out_ptr, params, plan,
//                          line_buffer, ext_full, y)
//
// The only difference between the public entry points is how the input
// samples are fed into the line buffer:
//
//   - resize_1d_line_contiguous: copies from a raw contiguous double* line
//   - resize_1d_workspace:       copies from a std::vector<double>
//   - resize_1d_line_buffered:   caller has already filled the line buffer

namespace lsresize {

namespace {

static inline std::uint64_t double_bits(double value)
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct PlanCacheKey {
  int N;
  int interp_degree;
  int analy_degree;
  int synthe_degree;
  std::uint64_t zoom_bits;
  std::uint64_t shift_bits;
  bool inversable;
};

static inline bool operator==(const PlanCacheKey& a, const PlanCacheKey& b)
{
  return a.N == b.N &&
         a.interp_degree == b.interp_degree &&
         a.analy_degree == b.analy_degree &&
         a.synthe_degree == b.synthe_degree &&
         a.zoom_bits == b.zoom_bits &&
         a.shift_bits == b.shift_bits &&
         a.inversable == b.inversable;
}

static inline PlanCacheKey plan_cache_key(int N, const LSParams& p)
{
  return PlanCacheKey{
      N,
      p.interp_degree,
      p.analy_degree,
      p.synthe_degree,
      double_bits(p.zoom),
      double_bits(p.shift),
      p.inversable};
}

static inline int plan_cache_capacity()
{
  if (const char* value = std::getenv("LSRESIZE_PLAN_CACHE_SIZE")) {
    const int parsed = std::atoi(value);
    return std::max(0, parsed);
  }
  return 32;
}

struct PlanCacheEntry {
  PlanCacheKey key;
  std::shared_ptr<const Plan1D> plan;
};

static std::mutex& plan_cache_mutex()
{
  static std::mutex m;
  return m;
}

static std::list<PlanCacheEntry>& plan_cache_entries()
{
  static std::list<PlanCacheEntry> entries;
  return entries;
}

static std::shared_ptr<const Plan1D> find_cached_plan_locked(
  const PlanCacheKey& key)
{
  auto& entries = plan_cache_entries();
  for (auto it = entries.begin(); it != entries.end(); ++it) {
    if (it->key == key) {
      auto plan = it->plan;
      entries.splice(entries.begin(), entries, it);
      return plan;
    }
  }
  return nullptr;
}

static void trim_plan_cache_locked(int capacity)
{
  auto& entries = plan_cache_entries();
  while (static_cast<int>(entries.size()) > capacity) {
    entries.pop_back();
  }
}

static inline void append_row_run(
  std::vector<RowRun1D>& runs,
  int row,
  bool interior)
{
  const char flag = interior ? 1 : 0;
  if (!runs.empty() &&
      runs.back().interior == flag &&
      runs.back().end == row) {
    runs.back().end = row + 1;
    return;
  }
  runs.push_back(RowRun1D{row, row + 1, flag});
}

static inline bool is_direct_interior_row(
  const Plan1D& plan,
  int row)
{
  const int begin = plan.row_ptr[static_cast<size_t>(row)];
  const int endw = plan.row_ptr[static_cast<size_t>(row) + 1];
  const int k0 = plan.kmin[static_cast<size_t>(row)];
  const int kmax = k0 + (endw - begin) - 1;
  return (begin == endw || (k0 >= 0 && kmax < plan.N));
}

static inline void mapped_coeff_col(
  int k,
  const Plan1D& plan,
  int& src,
  double& sgn)
{
  const int N = plan.N;
  if (k < 0) {
    if (plan.left_pad <= 0 || plan.pad_src_idx.empty()) {
      src = 0;
      sgn = 1.0;
      return;
    }
    const int pad_i = std::min(
        std::max(plan.left_pad + k, 0),
        std::max(0, plan.left_pad - 1));
    src = std::min(
        std::max(plan.pad_src_idx[static_cast<size_t>(pad_i)], 0),
        std::max(0, N - 1));
    sgn = static_cast<int>(plan.pad_src_sgn[static_cast<size_t>(pad_i)]);
    return;
  }

  if (k < N) {
    src = k;
    sgn = 1.0;
    return;
  }

  const int last_k = std::max(0, plan.length_total - 1);
  const int ext_k = (k < plan.length_total) ? k : last_k;
  if (ext_k < N || plan.rp_src.empty()) {
    src = std::min(std::max(ext_k, 0), std::max(0, N - 1));
    sgn = 1.0;
    return;
  }

  const int tail_i = std::min(
      std::max(ext_k - N, 0),
      static_cast<int>(plan.rp_src.size()) - 1);
  src = std::min(
      std::max(plan.rp_src[static_cast<size_t>(tail_i)], 0),
      std::max(0, N - 1));
  sgn = static_cast<int>(plan.rp_sign);
}

static void precompute_batched_row_map(Plan1D& plan)
{
  plan.row_runs.clear();
  plan.coeff_src.assign(plan.weights.size(), 0);
  plan.coeff_sgn.assign(plan.weights.size(), 1.0);
  plan.interior_rows = 0;
  plan.mapped_rows = 0;
  plan.row_runs.reserve(static_cast<size_t>(3));

  for (int l = 0; l < plan.out_total; ++l) {
    const int begin = plan.row_ptr[static_cast<size_t>(l)];
    const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
    const int k0 = plan.kmin[static_cast<size_t>(l)];
    const bool interior = is_direct_interior_row(plan, l);
    append_row_run(plan.row_runs, l, interior);

    if (interior) {
      ++plan.interior_rows;
      continue;
    }

    ++plan.mapped_rows;
    for (int t = begin; t < endw; ++t) {
      const size_t ti = static_cast<size_t>(t);
      const int kt = k0 + (t - begin);
      mapped_coeff_col(kt, plan, plan.coeff_src[ti], plan.coeff_sgn[ti]);
    }
  }
}

static void precompute_direct_linear_map(Plan1D& plan, const LSParams& p)
{
  plan.direct_linear_ok = false;
  plan.direct_linear_count.clear();
  plan.direct_linear_src0.clear();
  plan.direct_linear_src1.clear();
  plan.direct_linear_src2.clear();
  plan.direct_linear_w0.clear();
  plan.direct_linear_w1.clear();
  plan.direct_linear_w2.clear();
  plan.direct_linear_w0_f32.clear();
  plan.direct_linear_w1_f32.clear();
  plan.direct_linear_w2_f32.clear();

  if (p.analy_degree >= 0 ||
      p.synthe_degree != p.interp_degree ||
      p.interp_degree != 1) {
    return;
  }

  const int outN = plan.outN;
  plan.direct_linear_count.assign(static_cast<size_t>(outN), 0);
  plan.direct_linear_src0.assign(static_cast<size_t>(outN), 0);
  plan.direct_linear_src1.assign(static_cast<size_t>(outN), 0);
  plan.direct_linear_src2.assign(static_cast<size_t>(outN), 0);
  plan.direct_linear_w0.assign(static_cast<size_t>(outN), 0.0);
  plan.direct_linear_w1.assign(static_cast<size_t>(outN), 0.0);
  plan.direct_linear_w2.assign(static_cast<size_t>(outN), 0.0);
  plan.direct_linear_w0_f32.assign(static_cast<size_t>(outN), 0.0f);
  plan.direct_linear_w1_f32.assign(static_cast<size_t>(outN), 0.0f);
  plan.direct_linear_w2_f32.assign(static_cast<size_t>(outN), 0.0f);

  const double* weights = plan.weights.data();
  const int* coeff_src = plan.coeff_src.data();
  const double* coeff_sgn = plan.coeff_sgn.data();

  for (int l = 0; l < outN; ++l) {
    const size_t li = static_cast<size_t>(l);
    const int begin = plan.row_ptr[li];
    const int endw = plan.row_ptr[li + 1];
    const int m = endw - begin;
    if (m < 0 || m > 3) {
      plan.direct_linear_count.clear();
      plan.direct_linear_src0.clear();
      plan.direct_linear_src1.clear();
      plan.direct_linear_src2.clear();
      plan.direct_linear_w0.clear();
      plan.direct_linear_w1.clear();
      plan.direct_linear_w2.clear();
      plan.direct_linear_w0_f32.clear();
      plan.direct_linear_w1_f32.clear();
      plan.direct_linear_w2_f32.clear();
      return;
    }

    plan.direct_linear_count[li] = static_cast<unsigned char>(m);
    const int k0 = plan.kmin[li];
    const int kmax = k0 + m - 1;
    const bool interior = (m == 0 || (k0 >= 0 && kmax < plan.N));

    int src[3] = {0, 0, 0};
    double w[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < m; ++j) {
      const int t = begin + j;
      const size_t ti = static_cast<size_t>(t);
      if (interior) {
        src[j] = k0 + j;
        w[j] = weights[ti];
      } else {
        src[j] = coeff_src[ti];
        w[j] = weights[ti] * coeff_sgn[ti];
      }
    }

    plan.direct_linear_src0[li] = src[0];
    plan.direct_linear_src1[li] = src[1];
    plan.direct_linear_src2[li] = src[2];
    plan.direct_linear_w0[li] = w[0];
    plan.direct_linear_w1[li] = w[1];
    plan.direct_linear_w2[li] = w[2];
    plan.direct_linear_w0_f32[li] = static_cast<float>(w[0]);
    plan.direct_linear_w1_f32[li] = static_cast<float>(w[1]);
    plan.direct_linear_w2_f32[li] = static_cast<float>(w[2]);
  }

  plan.direct_linear_ok = true;
}

} // namespace

// Build the reusable 1-D plan (window metadata + contiguous weights + pad map)
Plan1D make_plan_1d(int N, const LSParams& p)
{
  Plan1D plan{};
  plan.N = N;

  // Output size (same as before)
  int workN = 0, outN = 0;
  calculate_final_size_1d(p.inversable, N, p.zoom, workN, outN);
  plan.outN = outN;

  const bool pure_interp = (p.analy_degree < 0);

  // total_degree controls the spline support used in the windows
  const int total_degree = p.interp_degree + p.analy_degree + 1;

  // Correction degree for LS / oblique projection
  const int corr_degree = pure_interp
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // Tail length / out_total
  //  - Pure interpolation: no projection tail, only outN samples
  //  - LS / oblique: keep original border-based tail
  int add_border = 0;
  if (!pure_interp) {
    add_border = std::max(border(outN, corr_degree), total_degree);
  }
  plan.out_total = outN + add_border;

  // Shift:
  //  - Interpolation uses p.shift as-is
  //  - Projection adds the Muñoz correction
  double shift = p.shift;
  if (!pure_interp) {
    const double t = (p.analy_degree + 1.0) / 2.0;
    shift += (t - std::floor(t)) * (1.0 / p.zoom - 1.0);
  }

  // Symmetric (even) vs antisymmetric (odd) boundary
  plan.symmetric_ext = ((p.analy_degree + 1) % 2 == 0);

  const double half_support = 0.5 * (total_degree + 1);

  // Zoom exponent for LS / oblique (Unser–Muñoz step 3 factor)
  const double fact = std::pow(
      p.zoom,
      (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0
  );

  // Extended input length:
  //  - Interpolation: only need a small mirror tail up to the spline support.
  //  - LS / oblique: original LS sizing using add_border/zoom.
  if (pure_interp) {
    const int right_ext = static_cast<int>(std::ceil(half_support));
    plan.length_total   = N + right_ext;
  } else {
    plan.length_total   = N + static_cast<int>(std::ceil(add_border / p.zoom));
  }

  // CSR-style window metadata
  plan.row_ptr.resize(static_cast<size_t>(plan.out_total) + 1);
  plan.kmin   .resize(static_cast<size_t>(plan.out_total));
  plan.win_len.resize(static_cast<size_t>(plan.out_total));

  int nnz      = 0;
  int min_kmin =  0;
  int max_kmax = -1;

  // Unified TensorSpline-style geometry for ALL methods:
  //
  //   - Input samples at k = 0 .. N-1
  //   - Visible outputs (0 .. outN-1) span [0, N-1]
  //     => step = (N-1)/(outN-1) when outN > 1
  //   - Tail samples (l >= outN) simply continue with the same step.
  const double step = (plan.outN > 1)
                    ? (static_cast<double>(N - 1) /
                       static_cast<double>(plan.outN - 1))
                    : 0.0;

  // First pass: compute (kmin, kmax) per row, nnz, global min/max
  for (int l = 0; l < plan.out_total; ++l) {
    const double x = step * static_cast<double>(l) + shift;

    const int kmin = static_cast<int>(std::ceil (x - half_support));
    const int kmax = static_cast<int>(std::floor(x + half_support));
    const int wlen = kmax - kmin + 1;

    plan.kmin   [static_cast<size_t>(l)] = kmin;
    plan.win_len[static_cast<size_t>(l)] = wlen;
    nnz += wlen;

    if (kmin < min_kmin) min_kmin = kmin;
    if (kmax > max_kmax) max_kmax = kmax;
  }

  // Global pads to build a single contiguous extended buffer: [LP | ext | RP]
  plan.left_pad  = std::max(0, -min_kmin);
  plan.right_pad = std::max(0,  max_kmax - (plan.length_total - 1));

  // Precompute left-pad mapping for negative indices: -t -> sign * line[src]
  plan.pad_src_idx.resize(static_cast<size_t>(plan.left_pad));
  plan.pad_src_sgn.resize(static_cast<size_t>(plan.left_pad), 1);
  for (int t = 1; t <= plan.left_pad; ++t) {
    const int pos = plan.left_pad - t; // 0 .. left_pad-1
    if (plan.symmetric_ext) {
      // symmetric: -t -> +line[t]
      plan.pad_src_idx[static_cast<size_t>(pos)] = t;   // clamped later to [0, N-1]
      plan.pad_src_sgn[static_cast<size_t>(pos)] =  1;
    } else {
      // antisymmetric: -t -> -line[t-1]
      plan.pad_src_idx[static_cast<size_t>(pos)] = t - 1;
      plan.pad_src_sgn[static_cast<size_t>(pos)] = -1;
    }
  }

  // Allocate contiguous weights (sign handled via extension, not weights)
  plan.weights.resize(static_cast<size_t>(nnz));

  // Second pass: fill row_ptr and weights
  int cursor = 0;
  for (int l = 0; l < plan.out_total; ++l) {
    plan.row_ptr[static_cast<size_t>(l)] = cursor;

    const double x = step * static_cast<double>(l) + shift;
    const int    k0   = plan.kmin   [static_cast<size_t>(l)];
    const int    wlen = plan.win_len[static_cast<size_t>(l)];

    for (int t = 0; t < wlen; ++t) {
      const int k = k0 + t;
      const double w = fact * beta(x - k, total_degree);
      plan.weights[static_cast<size_t>(cursor++)] = w;
    }
  }
  plan.row_ptr.back() = cursor;

  // --- Precompute right extension mapping (mirrored indices) ---
  {
    const int rem = plan.length_total - N;
    plan.rp_src.clear();
    plan.rp_sign = plan.symmetric_ext ?  1 : -1;

    if (rem > 0) {
      plan.rp_src.resize(static_cast<size_t>(rem));
      if (plan.symmetric_ext) {
        const int period = 2 * N - 2;
        for (int l = N; l < plan.length_total; ++l) {
          int t = l;
          if (period > 0 && t >= period) t %= period;
          if (t >= N) t = period - t;
          if (t < 0) t = 0; else if (t >= N) t = N - 1;
          plan.rp_src[static_cast<size_t>(l - N)] = t;
        }
      } else { // antisymmetric
        const int period = 2 * N - 3;
        for (int l = N; l < plan.length_total; ++l) {
          int t = l;
          if (period > 0 && t >= period) t %= period;
          if (t >= N) t = period - t;
          if (t < 0) t = 0; else if (t >= N) t = N - 1;
          plan.rp_src[static_cast<size_t>(l - N)] = t;
        }
      }
    }
  }

  precompute_batched_row_map(plan);
  precompute_direct_linear_map(plan, p);

  return plan;
}

std::shared_ptr<const Plan1D> get_plan_1d_cached(int N, const LSParams& p)
{
  const int capacity = plan_cache_capacity();
  if (capacity <= 0) {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::PlanBuild);
    return std::make_shared<Plan1D>(make_plan_1d(N, p));
  }

  const PlanCacheKey key = plan_cache_key(N, p);
  {
    std::lock_guard<std::mutex> lock(plan_cache_mutex());
    if (auto plan = find_cached_plan_locked(key)) {
      trim_plan_cache_locked(capacity);
      return plan;
    }
  }

  std::shared_ptr<const Plan1D> built;
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::PlanBuild);
    built = std::make_shared<Plan1D>(make_plan_1d(N, p));
  }

  {
    std::lock_guard<std::mutex> lock(plan_cache_mutex());
    if (auto plan = find_cached_plan_locked(key)) {
      trim_plan_cache_locked(capacity);
      return plan;
    }
    auto& entries = plan_cache_entries();
    entries.push_front(PlanCacheEntry{key, built});
    trim_plan_cache_locked(capacity);
  }

  return built;
}

// -----------------------------------------------------------------------------
// Internal 1-D cores
// -----------------------------------------------------------------------------

// Core that assumes `line[0..N-1]` is already filled with input samples.
// This lets callers (like the ND kernel) avoid an extra copy into line.
static inline void run_pipeline_from_line(
  double* out,
  const LSParams& p,
  const Plan1D& plan,
  std::vector<double>& line,
  std::vector<double>& ext_full,
  std::vector<double>& y)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DTotal);

  const int N = plan.N;
  if (N == 0) {
    return;
  }

  const int corr_degree = (p.analy_degree < 0)
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // 1) Interpolation coefficients (causal/anti-causal IIR on input), in-place.
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DPrefilter);
    get_interpolation_coefficients(line, p.interp_degree);
  }

  // 2) Optional projection integration
  double average = 0.0;
  if (p.analy_degree >= 0) {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DIntegrate);
    average = do_integ(line, p.analy_degree + 1);
  }

  // 3) Single padded buffer [LP | ext | RP], built directly from line
  const int LP     = plan.left_pad;
  const int length = plan.length_total;
  const int RP     = plan.right_pad;

  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DExtend);
    ext_full.resize(static_cast<size_t>(LP + length + RP));
    double* dst = ext_full.data();

    // 3a) Left pad using the precomputed mapping
    if (LP > 0) {
      for (int i = 0; i < LP; ++i) {
        const int src = std::min(
            std::max(plan.pad_src_idx[static_cast<size_t>(i)], 0),
            std::max(0, N - 1));
        const int sgn = static_cast<int>(plan.pad_src_sgn[static_cast<size_t>(i)]);
        dst[static_cast<size_t>(i)] = sgn * line[static_cast<size_t>(src)];
      }
    }

    // 3b) Main input samples
    std::copy(line.begin(), line.end(), dst + LP);

    // 3c) Right extension into the middle block
    const int rem = length - N;
    if (rem > 0 && !plan.rp_src.empty()) {
      const double sgn = static_cast<int>(plan.rp_sign);
      double* tail = dst + LP + N;
      for (int i = 0; i < rem; ++i) {
        const int src = plan.rp_src[static_cast<size_t>(i)];
        tail[static_cast<size_t>(i)] = sgn * line[static_cast<size_t>(src)];
      }
    }

    // 3d) Right pad (clamp to last sample)
    if (RP > 0) {
      const double last = dst[LP + length - 1];
      std::fill(dst + LP + length, dst + LP + length + RP, last);
    }
  }

  // 4) Accumulate using the plan (contiguous weights & samples)
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DAccumulate);
    y.resize(static_cast<size_t>(plan.out_total));
    const int*    rp = plan.row_ptr.data();
    const double* ww = plan.weights.data();
    const double* vf = ext_full.data();

    for (int l = 0; l < plan.out_total; ++l) {
      const int begin = rp[static_cast<size_t>(l)];
      const int end   = rp[static_cast<size_t>(l) + 1];
      const int M     = end - begin;
      const int k0    = plan.kmin[static_cast<size_t>(l)];

      const double* w = ww + begin;
      const double* v = vf + (LP + k0);

      y[static_cast<size_t>(l)] = dot_small(w, v, M);
    }
  }

  // 5) Projection tail (unchanged)
  if (p.analy_degree >= 0) {
    {
      LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DDiff);
      do_diff(y, p.analy_degree + 1);
      for (int i = 0; i < plan.out_total; ++i) {
        y[static_cast<size_t>(i)] += average;
      }
    }
    {
      LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DOutputPrefilter);
      get_interpolation_coefficients(y, corr_degree);
    }
    {
      LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DSampling);
      get_samples(y, p.synthe_degree);
    }
  }

  // 6) Copy to true output size
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DOutputCopy);
    std::copy(y.begin(), y.begin() + plan.outN, out);
  }
}

// Raw-pointer core: operates directly on in/out buffers using workspace vectors.
// This version still accepts `in` and copies it into `line` once, then
// delegates to run_pipeline_from_line.
static inline void run_pipeline_from_raw(
  const double* in_samples,
  double* out_samples,
  const LSParams& p,
  const Plan1D& plan,
  std::vector<double>& line,
  std::vector<double>& ext_full,
  std::vector<double>& y)
{
  const int N = plan.N;
  if (N == 0) return;

  line.resize(static_cast<size_t>(N));
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DRawCopy);
    std::copy(in_samples, in_samples + N, line.begin());
  }

  run_pipeline_from_line(out_samples, p, plan,
                         line, ext_full, y);
}

// Old vector API now just wraps the raw core.
static inline void resize_1d_core(
  const std::vector<double>& in,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  std::vector<double>& line,
  std::vector<double>& ext_full,
  std::vector<double>& y)
{
  const int N = plan.N;
  if (N == 0) {
    out.clear();
    return;
  }
  out.resize(static_cast<size_t>(plan.outN));
  run_pipeline_from_raw(in.data(), out.data(), p, plan, line, ext_full, y);
}

// -----------------------------------------------------------------------------
// Public, allocation-free wrappers
// -----------------------------------------------------------------------------

void resize_1d_workspace(
  const std::vector<double>& in,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace)
{
  resize_1d_core(in, out, p, plan, workspace.line, workspace.ext_full, workspace.y);
}

// Raw-pointer wrapper for contiguous lines (no std::vector in/out).
void resize_1d_line_contiguous(
  const double* in,
  double* out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace)
{
  run_pipeline_from_raw(in, out, p, plan, workspace.line, workspace.ext_full, workspace.y);
}

// Wrapper used by the ND kernel when it has already filled `line`.
// Avoids an extra copy from a temporary into workspace.line.
void resize_1d_line_buffered(
  std::vector<double>& line,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace)
{
  const int N = plan.N;
  if (N == 0) {
    out.clear();
    return;
  }
  out.resize(static_cast<size_t>(plan.outN));
  run_pipeline_from_line(out.data(), p, plan, line, workspace.ext_full, workspace.y);
}

} // namespace lsresize
