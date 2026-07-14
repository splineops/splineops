// splineops/cpp/lsresize/src/resize_1d.cpp
#include "resize_1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"
#include "dot_kernels.h"
#include "profile_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#endif

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
  int outN;
  int interp_degree;
  int analy_degree;
  int synthe_degree;
  std::uint64_t shift_bits;
};

static inline bool operator==(const PlanCacheKey& a, const PlanCacheKey& b)
{
  return a.N == b.N &&
         a.outN == b.outN &&
         a.interp_degree == b.interp_degree &&
         a.analy_degree == b.analy_degree &&
         a.synthe_degree == b.synthe_degree &&
         a.shift_bits == b.shift_bits;
}

static inline PlanCacheKey plan_cache_key(int N, const LSParams& p)
{
  const int outN = calculate_output_size_1d(N, p.zoom);
  return PlanCacheKey{
      N,
      outN,
      p.interp_degree,
      p.analy_degree,
      p.synthe_degree,
      double_bits(p.shift)};
}

constexpr std::size_t kDefaultPlanCacheEntries = 32;
constexpr std::size_t kDefaultPlanCacheBytes =
    std::size_t{128} * 1024 * 1024;

static inline bool ascii_space(char value) noexcept
{
  return value == ' ' || value == '\t' || value == '\n' ||
         value == '\r' || value == '\f' || value == '\v';
}

static bool parse_nonnegative_size(
  const char* text,
  std::size_t& parsed) noexcept
{
  if (text == nullptr) {
    return false;
  }
  while (ascii_space(*text)) {
    ++text;
  }
  if (*text == '+') {
    ++text;
  }

  bool saw_digit = false;
  std::size_t value = 0;
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  for (; *text >= '0' && *text <= '9'; ++text) {
    saw_digit = true;
    const std::size_t digit = static_cast<std::size_t>(*text - '0');
    if (value > maximum / 10 ||
        (value == maximum / 10 && digit > maximum % 10)) {
      return false;
    }
    value = value * 10 + digit;
  }
  while (ascii_space(*text)) {
    ++text;
  }
  if (!saw_digit || *text != '\0') {
    return false;
  }
  parsed = value;
  return true;
}

static std::size_t plan_cache_limit(
  const char* name,
  std::size_t fallback) noexcept
{
  const char* text = std::getenv(name);
  if (text == nullptr) {
    return fallback;
  }
  std::size_t parsed = 0;
  return parse_nonnegative_size(text, parsed) ? parsed : fallback;
}

struct PlanCacheLimits {
  std::size_t entries;
  std::size_t bytes;

  bool enabled() const noexcept
  {
    return entries > 0 && bytes > 0;
  }
};

static PlanCacheLimits plan_cache_limits() noexcept
{
  return PlanCacheLimits{
      plan_cache_limit(
          "LSRESIZE_PLAN_CACHE_SIZE", kDefaultPlanCacheEntries),
      plan_cache_limit(
          "LSRESIZE_PLAN_CACHE_BYTES", kDefaultPlanCacheBytes)};
}

static void add_memory_bytes(
  std::size_t& total,
  std::size_t count,
  std::size_t element_size) noexcept
{
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (count > maximum / element_size) {
    total = maximum;
    return;
  }
  const std::size_t bytes = count * element_size;
  total = (bytes > maximum - total) ? maximum : total + bytes;
}

template <typename T>
static void add_vector_capacity(
  std::size_t& total,
  const std::vector<T>& values) noexcept
{
  add_memory_bytes(total, values.capacity(), sizeof(T));
}

static std::size_t plan_memory_bytes(const Plan1D& plan) noexcept
{
  // sizeof(Plan1D) includes every vector control block. The fixed allowance
  // conservatively covers the shared_ptr control block, list node, allocator
  // bookkeeping, and alignment beyond the vector capacities below.
  std::size_t total = sizeof(Plan1D) + 256;
  add_vector_capacity(total, plan.row_ptr);
  add_vector_capacity(total, plan.weights);
  add_vector_capacity(total, plan.kmin);
  add_vector_capacity(total, plan.win_len);
  add_vector_capacity(total, plan.pad_src_idx);
  add_vector_capacity(total, plan.pad_src_sgn);
  add_vector_capacity(total, plan.rp_src);
  add_vector_capacity(total, plan.row_runs);
  add_vector_capacity(total, plan.coeff_src);
  add_vector_capacity(total, plan.coeff_sgn);
  add_vector_capacity(total, plan.direct_linear_count);
  add_vector_capacity(total, plan.direct_linear_src0);
  add_vector_capacity(total, plan.direct_linear_src1);
  add_vector_capacity(total, plan.direct_linear_src2);
  add_vector_capacity(total, plan.direct_linear_w0);
  add_vector_capacity(total, plan.direct_linear_w1);
  add_vector_capacity(total, plan.direct_linear_w2);
  add_vector_capacity(total, plan.direct_linear_w0_f32);
  add_vector_capacity(total, plan.direct_linear_w1_f32);
  add_vector_capacity(total, plan.direct_linear_w2_f32);
  return total;
}

struct PlanCacheEntry {
  PlanCacheKey key;
  std::shared_ptr<const Plan1D> plan;
  std::size_t bytes;
};

class PlanCacheState;

#if !defined(_WIN32)
static PlanCacheState*& active_plan_cache_state() noexcept
{
  static PlanCacheState* state = nullptr;
  return state;
}
#endif

class PlanCacheState {
public:
  PlanCacheState()
  {
#if !defined(_WIN32)
    active_plan_cache_state() = this;
    fork_safe_ = (::pthread_atfork(
        &PlanCacheState::prepare_fork,
        &PlanCacheState::parent_after_fork,
        &PlanCacheState::child_after_fork) == 0);
#endif
  }

  ~PlanCacheState()
  {
#if !defined(_WIN32)
    active_plan_cache_state() = nullptr;
#endif
  }

  PlanCacheState(const PlanCacheState&) = delete;
  PlanCacheState& operator=(const PlanCacheState&) = delete;

  bool usable() const noexcept
  {
    return fork_safe_;
  }

  std::mutex mutex;
  std::list<PlanCacheEntry> entries;
  std::size_t bytes = 0;

private:
#if !defined(_WIN32)
  static void prepare_fork() noexcept
  {
    if (PlanCacheState* state = active_plan_cache_state()) {
      state->mutex.lock();
    }
  }

  static void parent_after_fork() noexcept
  {
    if (PlanCacheState* state = active_plan_cache_state()) {
      state->mutex.unlock();
    }
  }

  static void child_after_fork() noexcept
  {
    if (PlanCacheState* state = active_plan_cache_state()) {
      // Keep the immutable inherited plans. Avoid allocations, destruction,
      // or refcount changes in the child handler; only release the mutex that
      // prepare_fork acquired in the forking thread.
      state->mutex.unlock();
    }
  }
#endif

  bool fork_safe_ = true;
};

static PlanCacheState& plan_cache_state()
{
  static PlanCacheState state;
  return state;
}

static std::shared_ptr<const Plan1D> find_cached_plan_locked(
  PlanCacheState& cache,
  const PlanCacheKey& key)
{
  for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
    if (it->key == key) {
      auto plan = it->plan;
      cache.entries.splice(cache.entries.begin(), cache.entries, it);
      return plan;
    }
  }
  return nullptr;
}

static void evict_plan_cache_locked(
  PlanCacheState& cache,
  const PlanCacheLimits& limits,
  std::list<PlanCacheEntry>& evicted)
{
  while (cache.entries.size() > limits.entries ||
         cache.bytes > limits.bytes) {
    if (cache.entries.empty()) {
      cache.bytes = 0;
      break;
    }
    auto last = std::prev(cache.entries.end());
    cache.bytes = (last->bytes <= cache.bytes)
        ? cache.bytes - last->bytes
        : 0;
    evicted.splice(evicted.begin(), cache.entries, last);
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
  std::int8_t& sgn)
{
  const int N = plan.N;
  if (plan.direct_projection) {
    const MirrorIndex mapped = mirror_index(
        static_cast<std::int64_t>(k), N, true);
    src = mapped.idx;
    sgn = static_cast<std::int8_t>(mapped.sign);
    return;
  }

  if (k < 0) {
    if (plan.left_pad <= 0 || plan.pad_src_idx.empty()) {
      src = 0;
      sgn = 1;
      return;
    }
    const int pad_i = std::min(
        std::max(plan.left_pad + k, 0),
        std::max(0, plan.left_pad - 1));
    src = std::min(
        std::max(plan.pad_src_idx[static_cast<size_t>(pad_i)], 0),
        std::max(0, N - 1));
    sgn = static_cast<std::int8_t>(
        plan.pad_src_sgn[static_cast<size_t>(pad_i)]);
    return;
  }

  if (k < N) {
    src = k;
    sgn = 1;
    return;
  }

  const int last_k = std::max(0, plan.length_total - 1);
  const int ext_k = (k < plan.length_total) ? k : last_k;
  if (ext_k < N || plan.rp_src.empty()) {
    src = std::min(std::max(ext_k, 0), std::max(0, N - 1));
    sgn = 1;
    return;
  }

  const int tail_i = std::min(
      std::max(ext_k - N, 0),
      static_cast<int>(plan.rp_src.size()) - 1);
  src = std::min(
      std::max(plan.rp_src[static_cast<size_t>(tail_i)], 0),
      std::max(0, N - 1));
  sgn = static_cast<std::int8_t>(plan.rp_sign);
}

static void precompute_batched_row_map(Plan1D& plan)
{
  plan.row_runs.clear();
  plan.coeff_src.assign(plan.weights.size(), 0);
  plan.coeff_sgn.assign(plan.weights.size(), std::int8_t{1});
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
  const std::int8_t* coeff_sgn = plan.coeff_sgn.data();

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

struct GaussRule1D {
  int count = 0;
  std::array<double, 4> nodes{};
  std::array<double, 4> weights{};
};

static GaussRule1D gauss_rule_1d(int order)
{
  if (order == 1) {
    return GaussRule1D{1, {0.0, 0.0, 0.0, 0.0},
                       {2.0, 0.0, 0.0, 0.0}};
  }
  if (order == 2) {
    const double x = 1.0 / std::sqrt(3.0);
    return GaussRule1D{2, {-x, x, 0.0, 0.0},
                       {1.0, 1.0, 0.0, 0.0}};
  }
  if (order == 3) {
    const double x = std::sqrt(3.0 / 5.0);
    return GaussRule1D{3, {-x, 0.0, x, 0.0},
                       {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0, 0.0}};
  }
  const double s = std::sqrt(6.0 / 5.0);
  const double x0 = std::sqrt((3.0 + 2.0 * s) / 7.0);
  const double x1 = std::sqrt((3.0 - 2.0 * s) / 7.0);
  const double w0 = (18.0 - std::sqrt(30.0)) / 36.0;
  const double w1 = (18.0 + std::sqrt(30.0)) / 36.0;
  return GaussRule1D{4, {-x0, -x1, x1, x0},
                     {w0, w1, w1, w0}};
}

// Compact cross-inner-product kernel for projection:
//
//   h_a(x) = integral beta_n(t/a) beta_m(x-t) dt.
//
// Knot splitting makes the integrand polynomial of degree n+m on each
// interval. The corresponding fixed Gauss rule is exact in exact arithmetic
// and avoids the cancellation of the equivalent truncated-power formula.
static double cross_gram(
  double x,
  double scale,
  int interp_degree,
  int analysis_degree)
{
  const double input_radius =
      0.5 * static_cast<double>(interp_degree + 1);
  const double analysis_radius =
      0.5 * static_cast<double>(analysis_degree + 1);
  const double lo = std::max(
      -scale * input_radius, x - analysis_radius);
  const double hi = std::min(
      +scale * input_radius, x + analysis_radius);
  if (!(lo < hi)) return 0.0;

  std::array<double, 12> knots{};
  int count = 0;
  knots[static_cast<size_t>(count++)] = lo;
  knots[static_cast<size_t>(count++)] = hi;
  for (int i = 0; i <= interp_degree + 1; ++i) {
    const double value =
        scale * (-input_radius + static_cast<double>(i));
    if (value > lo && value < hi) {
      knots[static_cast<size_t>(count++)] = value;
    }
  }
  for (int j = 0; j <= analysis_degree + 1; ++j) {
    const double value =
        x - (-analysis_radius + static_cast<double>(j));
    if (value > lo && value < hi) {
      knots[static_cast<size_t>(count++)] = value;
    }
  }
  std::sort(knots.begin(), knots.begin() + count);
  int unique = 1;
  for (int i = 1; i < count; ++i) {
    if (knots[static_cast<size_t>(i)] !=
        knots[static_cast<size_t>(unique - 1)]) {
      knots[static_cast<size_t>(unique++)] = knots[static_cast<size_t>(i)];
    }
  }

  const GaussRule1D rule = gauss_rule_1d(
      (interp_degree + analysis_degree + 2) / 2);
  double total = 0.0;
  double correction = 0.0;
  for (int interval = 0; interval + 1 < unique; ++interval) {
    const double left = knots[static_cast<size_t>(interval)];
    const double right = knots[static_cast<size_t>(interval + 1)];
    const double half = 0.5 * (right - left);
    const double midpoint = 0.5 * (right + left);
    for (int q = 0; q < rule.count; ++q) {
      const double t = midpoint + half * rule.nodes[static_cast<size_t>(q)];
      const double term = half * rule.weights[static_cast<size_t>(q)] *
          beta(t / scale, interp_degree) *
          beta(x - t, analysis_degree);
      const double summed = total + term;
      correction += (std::abs(total) >= std::abs(term))
                  ? ((total - summed) + term)
                  : ((term - summed) + total);
      total = summed;
    }
  }
  const double value = total + correction;
  return (value < 0.0 && value > -64.0e-16) ? 0.0 : value;
}

static int checked_rounded_index(double value, const char* what)
{
  constexpr double int_min =
      static_cast<double>(std::numeric_limits<int>::min());
  constexpr double int_max =
      static_cast<double>(std::numeric_limits<int>::max());
  if (!std::isfinite(value) || value < int_min || value > int_max) {
    throw std::overflow_error(what);
  }
  return static_cast<int>(value);
}

static int checked_ceil_index(double value, const char* what)
{
  return checked_rounded_index(std::ceil(value), what);
}

static int checked_floor_index(double value, const char* what)
{
  return checked_rounded_index(std::floor(value), what);
}

static int checked_axis_add(int first, int second, const char* what)
{
  const std::int64_t result = static_cast<std::int64_t>(first) + second;
  if (first < 0 || second < 0 ||
      result > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(what);
  }
  return static_cast<int>(result);
}

} // namespace

// Build the reusable 1-D plan (window metadata + contiguous weights + pad map)
Plan1D make_plan_1d(int N, const LSParams& p)
{
  Plan1D plan{};
  plan.N = N;

  const int outN = calculate_output_size_1d(N, p.zoom);
  plan.outN = outN;

  const bool singleton_input = (N == 1);
  const bool singleton_output = (outN == 1);
  if (!singleton_input && !singleton_output) {
    plan.effective_zoom = endpoint_aligned_scale(N, outN);
  }

  const bool pure_interp = (p.analy_degree < 0);
  const bool visible_projection =
      !pure_interp && p.shift == 0.0 &&
      !singleton_input && !singleton_output;
  plan.direct_projection =
      visible_projection && p.analy_degree >= 1;

  // total_degree controls the spline support used in the windows
  const int total_degree = p.interp_degree + p.analy_degree + 1;

  // Correction degree for LS / oblique projection
  const int corr_degree = pure_interp
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // The zero-shift endpoint grid has its mirror boundary at outN-1. Its
  // boundary-aware differences and output Gram inverse therefore operate on
  // the visible sequence itself. Retain the historical tail only for the
  // internal non-zero-shift path, where endpoint symmetry does not apply.
  int add_border = 0;
  if (!pure_interp && !visible_projection) {
    add_border = std::max(border(outN, corr_degree), total_degree);
  }
  plan.out_total = checked_axis_add(
      outN, add_border, "projection output length exceeds native limits");

  // Degenerate endpoint grids have deliberately explicit semantics rather
  // than an invented scale: a singleton input is replicated, while a
  // projected singleton output is the line mean. The pipeline handles those
  // operations directly and does not need interpolation-window metadata.
  if (singleton_input || (singleton_output && !pure_interp)) {
    plan.out_total = outN;
    plan.length_total = N;
    plan.symmetric_ext = true;
    return plan;
  }

  // Shift:
  //  - Interpolation uses p.shift as-is
  //  - Projection adds the Muñoz correction
  double shift = p.shift;
  if (!pure_interp) {
    const double t = (p.analy_degree + 1.0) / 2.0;
    shift += (t - std::floor(t)) *
             (1.0 / plan.effective_zoom - 1.0);
  }

  // Direct cross-Gram rows consume the original symmetric interpolation
  // coefficients. Finite-difference rows consume the alternating symmetry
  // produced by their running sums.
  plan.symmetric_ext = plan.direct_projection
                     ? true
                     : ((p.analy_degree + 1) % 2 == 0);

  const double half_support = 0.5 * (total_degree + 1);

  // Zoom exponent for LS / oblique (Unser–Muñoz step 3 factor)
  const double fact = std::pow(
      plan.effective_zoom,
      (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0
  );

  // Extended input length. Endpoint-visible projection needs only enough
  // input continuation for its final support window; no output-IIR tail.
  if (pure_interp) {
    const int right_ext = checked_ceil_index(
        std::max(0.0, shift + half_support),
        "interpolation extension exceeds native limits");
    plan.length_total = checked_axis_add(
        N, right_ext, "interpolation extension exceeds native limits");
  } else if (visible_projection) {
    if (plan.direct_projection) {
      // Direct rows map every unwrapped coefficient index through the exact
      // whole-sample-symmetric period.  They therefore need no materialized
      // extension, even when a two-sample output spans several mirror periods.
      plan.length_total = N;
    } else {
      const int right_ext = checked_ceil_index(
          std::max(0.0, shift + half_support),
          "projection extension exceeds native limits");
      plan.length_total = checked_axis_add(
          N, right_ext, "projection extension exceeds native limits");
    }
  } else {
    const int right_ext = checked_ceil_index(
        add_border / plan.effective_zoom,
        "projection extension exceeds native limits");
    plan.length_total = checked_axis_add(
        N, right_ext, "projection extension exceeds native limits");
  }

  // CSR-style window metadata
  plan.row_ptr.resize(static_cast<size_t>(plan.out_total) + size_t{1});
  plan.kmin   .resize(static_cast<size_t>(plan.out_total));
  plan.win_len.resize(static_cast<size_t>(plan.out_total));

  std::size_t nnz = 0;
  int min_kmin =  0;
  int max_kmax = -1;

  // Unified TensorSpline-style geometry for ALL methods:
  //
  //   - Input samples at k = 0 .. N-1
  //   - Visible outputs (0 .. outN-1) span [0, N-1]
  //     => step = (N-1)/(outN-1) when outN > 1
  //   - Tail samples (l >= outN) simply continue with the same step.
  const double step = (plan.outN > 1)
                    ? (1.0 / plan.effective_zoom)
                    : 0.0;
  const double origin = (plan.outN == 1)
                      ? 0.5 * static_cast<double>(N - 1)
                      : 0.0;

  const double cross_radius = plan.direct_projection
      ? 0.5 * (
            (p.interp_degree + 1) * plan.effective_zoom +
            p.analy_degree + 1.0)
      : 0.0;

  // First pass: compute (kmin, kmax) per row, nnz, global min/max
  for (int l = 0; l < plan.out_total; ++l) {
    const double x = origin + step * static_cast<double>(l) + shift;
    const int kmin = plan.direct_projection
        ? checked_ceil_index(
              (static_cast<double>(l) - cross_radius) /
                  plan.effective_zoom,
              "projection source index exceeds native limits")
        : checked_ceil_index(
              x - half_support,
              "resampling source index exceeds native limits");
    const int kmax = plan.direct_projection
        ? checked_floor_index(
              (static_cast<double>(l) + cross_radius) /
                  plan.effective_zoom,
              "projection source index exceeds native limits")
        : checked_floor_index(
              x + half_support,
              "resampling source index exceeds native limits");
    const std::int64_t wlen64 =
        static_cast<std::int64_t>(kmax) - kmin + 1;
    if (wlen64 <= 0 ||
        wlen64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("resampling support width exceeds native limits");
    }
    const int wlen = static_cast<int>(wlen64);
    const std::size_t remaining =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) - nnz;
    if (static_cast<std::size_t>(wlen) > remaining) {
      throw std::overflow_error("resampling plan has too many nonzero weights");
    }

    plan.kmin   [static_cast<size_t>(l)] = kmin;
    plan.win_len[static_cast<size_t>(l)] = wlen;
    nnz += wlen;

    if (kmin < min_kmin) min_kmin = kmin;
    if (kmax > max_kmax) max_kmax = kmax;
  }

  // Global pads to build a single contiguous extended buffer: [LP | ext | RP]
  if (plan.direct_projection) {
    plan.left_pad = 0;
    plan.right_pad = 0;
  } else {
    const std::int64_t left_pad = std::max<std::int64_t>(
        0, -static_cast<std::int64_t>(min_kmin));
    const std::int64_t right_pad = std::max<std::int64_t>(
        0,
        static_cast<std::int64_t>(max_kmax) -
            (static_cast<std::int64_t>(plan.length_total) - 1));
    const std::int64_t full_length =
        left_pad + plan.length_total + right_pad;
    if (left_pad > std::numeric_limits<int>::max() ||
        right_pad > std::numeric_limits<int>::max() ||
        full_length > std::numeric_limits<int>::max()) {
      throw std::overflow_error("resampling extension exceeds native limits");
    }
    plan.left_pad = static_cast<int>(left_pad);
    plan.right_pad = static_cast<int>(right_pad);
  }

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
  plan.weights.resize(nnz);

  // Second pass: fill row_ptr and weights
  int cursor = 0;
  for (int l = 0; l < plan.out_total; ++l) {
    plan.row_ptr[static_cast<size_t>(l)] = cursor;

    const double x = origin + step * static_cast<double>(l) + shift;
    const int    k0   = plan.kmin   [static_cast<size_t>(l)];
    const int    wlen = plan.win_len[static_cast<size_t>(l)];

    const int row_begin = cursor;
    double row_sum = 0.0;
    for (int t = 0; t < wlen; ++t) {
      const int k = k0 + t;
      const double w = plan.direct_projection
          ? cross_gram(
                static_cast<double>(l) - plan.effective_zoom * k,
                plan.effective_zoom,
                p.interp_degree,
                p.analy_degree)
          : fact * beta(x - k, total_degree);
      plan.weights[static_cast<size_t>(cursor++)] = w;
      row_sum += w;
    }
    if (plan.direct_projection) {
      if (!(row_sum > 0.0) || !std::isfinite(row_sum)) {
        throw std::runtime_error(
            "direct projection cross-Gram row has zero weight");
      }
      const double inverse_sum = 1.0 / row_sum;
      for (int t = row_begin; t < cursor; ++t) {
        plan.weights[static_cast<size_t>(t)] *= inverse_sum;
      }
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
        for (int l = N; l < plan.length_total; ++l) {
          plan.rp_src[static_cast<size_t>(l - N)] =
              mirror_index(static_cast<std::int64_t>(l), N, true).idx;
        }
      } else { // antisymmetric
        const std::int64_t period =
            2 * static_cast<std::int64_t>(N) - 3;
        for (int l = N; l < plan.length_total; ++l) {
          std::int64_t t = l;
          if (period > 0 && t >= period) t %= period;
          if (t >= N) t = period - t;
          if (t < 0) t = 0; else if (t >= N) t = N - 1;
          plan.rp_src[static_cast<size_t>(l - N)] = static_cast<int>(t);
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
  PlanCacheState& cache = plan_cache_state();
  if (!cache.usable()) {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::PlanBuild);
    return std::make_shared<Plan1D>(make_plan_1d(N, p));
  }

  const PlanCacheKey key = plan_cache_key(N, p);
  // Splice evictions here so large vector allocations are released after the
  // cache mutex, rather than extending the critical section.
  std::list<PlanCacheEntry> evicted;
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const PlanCacheLimits limits = plan_cache_limits();
    evict_plan_cache_locked(cache, limits, evicted);
    if (limits.enabled()) {
      if (auto plan = find_cached_plan_locked(cache, key)) {
        return plan;
      }
    }
  }

  std::shared_ptr<const Plan1D> built;
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::PlanBuild);
    built = std::make_shared<Plan1D>(make_plan_1d(N, p));
  }
  const std::size_t built_bytes = plan_memory_bytes(*built);

  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const PlanCacheLimits limits = plan_cache_limits();
    evict_plan_cache_locked(cache, limits, evicted);
    if (!limits.enabled() || built_bytes > limits.bytes) {
      return built;
    }
    if (auto plan = find_cached_plan_locked(cache, key)) {
      return plan;
    }
    cache.entries.push_front(PlanCacheEntry{key, built, built_bytes});
    add_memory_bytes(cache.bytes, built_bytes, 1);
    evict_plan_cache_locked(cache, limits, evicted);
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

  if (N == 1) {
    std::fill(out, out + plan.outN, line[0]);
    return;
  }

  if (plan.outN == 1 && p.analy_degree >= 0) {
    long double sum = 0.0L;
    for (double value : line) {
      sum += static_cast<long double>(value);
    }
    out[0] = static_cast<double>(sum / static_cast<long double>(N));
    return;
  }

  if (plan.outN == 1 && p.analy_degree < 0 &&
      p.interp_degree == 0 && std::abs(p.shift) <= 1e-12) {
    const int left = (N - 1) / 2;
    const int right = N / 2;
    out[0] = 0.5 * (line[static_cast<size_t>(left)] +
                    line[static_cast<size_t>(right)]);
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
  if (p.analy_degree >= 0 && !plan.direct_projection) {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DIntegrate);
    average = do_integ(line, p.analy_degree + 1);
  }

  // 3) Single padded buffer [LP | ext | RP], built directly from line
  const int LP     = plan.left_pad;
  const int length = plan.length_total;
  const int RP     = plan.right_pad;

  if (!plan.direct_projection) {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::Pipeline1DExtend);
    ext_full.resize(
        static_cast<size_t>(LP) +
        static_cast<size_t>(length) +
        static_cast<size_t>(RP));
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
    const double* vf = plan.direct_projection ? nullptr : ext_full.data();

    for (int l = 0; l < plan.out_total; ++l) {
      const int begin = rp[static_cast<size_t>(l)];
      const int end   = rp[static_cast<size_t>(l) + 1];
      const int M     = end - begin;
      const int k0    = plan.kmin[static_cast<size_t>(l)];
      const double* w = ww + begin;

      if (!plan.direct_projection || is_direct_interior_row(plan, l)) {
        const double* v = plan.direct_projection
            ? line.data() + k0
            : vf + (LP + k0);
        y[static_cast<size_t>(l)] = dot_small(w, v, M);
        continue;
      }

      // Boundary direct-projection rows can span more than one mirror period under a
      // strong reduction. Their precomputed map is exact for every unwrapped
      // coefficient index and avoids allocating an O(N) padding buffer.
      double value = 0.0;
      for (int t = begin; t < end; ++t) {
        const size_t ti = static_cast<size_t>(t);
        value += ww[ti] * plan.coeff_sgn[ti] *
                 line[static_cast<size_t>(plan.coeff_src[ti])];
      }
      y[static_cast<size_t>(l)] = value;
    }
  }

  // 5) Projection tail (unchanged)
  if (p.analy_degree >= 0) {
    if (!plan.direct_projection) {
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
