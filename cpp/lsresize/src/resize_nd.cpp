// splineops/cpp/lsresize/src/resize_nd.cpp
#include "resize_nd.h"
#include "utils.h"
#include "parallel_utils.h"
#include "resize_1d.h"
#include "filters.h"
#include "profile_utils.h"

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>
#include <cmath>     // std::abs
#include <cstdlib>   // std::getenv, std::atof, std::atoi
#include <cstring>   // std::memcpy
#include <type_traits>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  #include <immintrin.h>
  #define LSRESIZE_GNU_X86_TARGETS 1
#else
  #define LSRESIZE_GNU_X86_TARGETS 0
#endif

namespace lsresize {

static std::vector<int64_t> strides_from_shape(
  const std::vector<int64_t>& shape) 
{
  std::vector<int64_t> s(shape.size(), 1);
  if (shape.empty()) return s;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    s[static_cast<size_t>(i)] =
        s[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return s;
}

static inline int64_t prod_elems(
  const std::vector<int64_t>& shape) 
{
  int64_t p = 1;
  for (int64_t v : shape) p *= v;
  return p;
}

static inline int64_t automatic_thread_cap_for_shape(
  const std::vector<int64_t>& shape)
{
  // Current v2 profiles show that small 3-D passes are memory/scheduler bound:
  // larger automatic pools add synchronization and cache traffic without
  // increasing useful throughput.  Keep this deliberately narrow; explicit
  // LSRESIZE_NUM_THREADS values continue to override the automatic ceiling.
  constexpr int64_t kSmall3DMaximumElements = 1000000;
  if (shape.size() == 3 && prod_elems(shape) <= kSmall3DMaximumElements) {
    return 8;
  }
  return static_cast<int64_t>(detail::kMaxParallelParticipants);
}

static inline bool is_small_3d_shape(const std::vector<int64_t>& shape)
{
  constexpr int64_t kSmall3DMaximumElements = 1000000;
  return shape.size() == 3 && prod_elems(shape) <= kSmall3DMaximumElements;
}

template <typename Worker>
static inline void run_parallel_for_shape(
  int64_t nlines,
  const Plan1D& plan,
  const std::vector<int64_t>& shape,
  Worker&& worker)
{
  run_parallel_or_serial(
      nlines,
      plan,
      std::forward<Worker>(worker),
      automatic_thread_cap_for_shape(shape),
      is_small_3d_shape(shape));
}

enum class BatchedAxisMode {
  Off,
  On,
  Auto
};

constexpr int kDefaultBatchLines = 64;

static inline char ascii_lower(char c)
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

static inline bool env_equals_ci(const char* value, const char* token)
{
  if (value == nullptr) return false;
  size_t i = 0;
  for (; token[i] != '\0'; ++i) {
    if (ascii_lower(value[i]) != token[i]) {
      return false;
    }
  }
  return value[i] == '\0';
}

static inline BatchedAxisMode batched_axis_mode()
{
  const char* value = std::getenv("LSRESIZE_BATCHED_AXIS");
  if (value == nullptr || value[0] == '\0') {
    return BatchedAxisMode::Auto;
  }
  if (env_equals_ci(value, "auto")) {
    return BatchedAxisMode::Auto;
  }
  if (value[0] == '0' ||
      env_equals_ci(value, "off") ||
      env_equals_ci(value, "false") ||
      env_equals_ci(value, "no")) {
    return BatchedAxisMode::Off;
  }
  if (value[0] == '1' ||
      value[0] == 't' ||
      value[0] == 'T' ||
      value[0] == 'y' ||
      value[0] == 'Y' ||
      env_equals_ci(value, "on") ||
      env_equals_ci(value, "true") ||
      env_equals_ci(value, "yes")) {
    return BatchedAxisMode::On;
  }
  return BatchedAxisMode::Off;
}

static inline int env_positive_int_or_zero(const char* name)
{
  if (const char* value = std::getenv(name)) {
    if (int parsed = std::atoi(value); parsed > 0) {
      return parsed;
    }
  }
  return 0;
}

static inline bool env_flag_enabled_default_true(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  if (value[0] == '0' ||
      env_equals_ci(value, "off") ||
      env_equals_ci(value, "false") ||
      env_equals_ci(value, "no")) {
    return false;
  }
  return true;
}

static inline bool specialized_presets_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_SPECIALIZED_PRESETS");
}

static inline bool row_gather_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_ROW_GATHER");
}

static inline bool strided_offset_gather_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_STRIDED_OFFSET_GATHER");
}

static inline bool gather_prefilter_scale_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_GATHER_PREFILTER_SCALE");
}

static inline bool projection_output_prefilter_scale_enabled_for(int64_t nlines)
{
  const char* value = std::getenv("LSRESIZE_PROJECTION_OUTPUT_PREFILTER_SCALE");
  if (value == nullptr || value[0] == '\0' || env_equals_ci(value, "auto")) {
    return explicit_thread_count(nlines) == 1;
  }
  if (value[0] == '0' ||
      env_equals_ci(value, "off") ||
      env_equals_ci(value, "false") ||
      env_equals_ci(value, "no")) {
    return false;
  }
  if (value[0] == '1' ||
      value[0] == 't' ||
      value[0] == 'T' ||
      value[0] == 'y' ||
      value[0] == 'Y' ||
      env_equals_ci(value, "on") ||
      env_equals_ci(value, "true") ||
      env_equals_ci(value, "yes")) {
    return true;
  }
  return false;
}

static inline bool projection_batch_tune_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_2D_PROJECTION_BATCH_TUNE");
}

static inline bool batch_tune_v2_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_BATCH_TUNE_V2");
}

static inline bool direct_3d_axis1_scatter_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_3D_AXIS1_DIRECT_SCATTER");
}

static inline bool direct_2d_axis0_scatter_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_2D_AXIS0_DIRECT_SCATTER");
}

static inline bool fast_linear_interp_enabled()
{
  const char* value = std::getenv("LSRESIZE_LINEAR_INTERP");
  if (value != nullptr && value[0] != '\0') {
    return env_flag_enabled_default_true("LSRESIZE_LINEAR_INTERP");
  }
  value = std::getenv("LSRESIZE_2D_LINEAR_INTERP");
  if (value != nullptr && value[0] != '\0') {
    return env_flag_enabled_default_true("LSRESIZE_2D_LINEAR_INTERP");
  }
  return env_flag_enabled_default_true("LSRESIZE_2D_FLOAT_INTERP");
}

static inline bool float32_internal_enabled()
{
  const char* value = std::getenv("LSRESIZE_PRECISION");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return env_equals_ci(value, "float32") ||
         env_equals_ci(value, "single") ||
         env_equals_ci(value, "f32");
}

static inline bool float32_internal_auto_enabled(
  const std::vector<int64_t>& in_shape,
  const LSParams& p)
{
  const char* value = std::getenv("LSRESIZE_PRECISION");
  if (value != nullptr && value[0] != '\0') {
    return false;
  }
  if (p.analy_degree < 0) {
    return (in_shape.size() == 2 || in_shape.size() == 3) &&
           p.synthe_degree == p.interp_degree &&
           (p.interp_degree == 2 || p.interp_degree == 3);
  }

  return in_shape.size() == 3 &&
         p.zoom < 1.0 &&
         ((p.interp_degree == 1 &&
           p.analy_degree == 0 &&
           p.synthe_degree == 1) ||
          (p.interp_degree == 2 &&
           p.analy_degree == 1 &&
           p.synthe_degree == 2) ||
          (p.interp_degree == 3 &&
           p.analy_degree == 1 &&
           p.synthe_degree == 3));
}

static inline bool float32_internal_enabled_for(
  const std::vector<int64_t>& in_shape,
  const LSParams& p)
{
  return float32_internal_enabled() ||
         float32_internal_auto_enabled(in_shape, p);
}

static inline bool avx2_linear_enabled()
{
#if LSRESIZE_GNU_X86_TARGETS
  if (!env_flag_enabled_default_true("LSRESIZE_AVX2_LINEAR")) {
    return false;
  }
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") &&
         __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

static inline bool last_axis_linear_direct_enabled()
{
  return env_flag_enabled_default_true("LSRESIZE_LAST_AXIS_LINEAR_DIRECT");
}

static inline bool fused_projection_average_restore_enabled_for(
  int64_t nlines,
  const Plan1D& plan)
{
  const char* value = std::getenv("LSRESIZE_FUSED_PROJECTION_AVG_RESTORE");
  if (value == nullptr || value[0] == '\0' || env_equals_ci(value, "auto")) {
    (void)plan;
    return explicit_thread_count(nlines) == 1;
  }
  if (value[0] == '0' ||
      env_equals_ci(value, "off") ||
      env_equals_ci(value, "false") ||
      env_equals_ci(value, "no")) {
    return false;
  }
  if (value[0] == '1' ||
      value[0] == 't' ||
      value[0] == 'T' ||
      value[0] == 'y' ||
      value[0] == 'Y' ||
      env_equals_ci(value, "on") ||
      env_equals_ci(value, "true") ||
      env_equals_ci(value, "yes")) {
    return true;
  }
  return false;
}

static inline int specialized_preset_max_support(
  const LSParams& p,
  const Plan1D& plan)
{
  // Direct cross-Gram support depends on the realized endpoint scale. The
  // fixed interpolation/FD widths below must never be used for those rows.
  if (plan.direct_projection) {
    return 0;
  }

  if (p.analy_degree < 0 && p.synthe_degree == p.interp_degree) {
    if (p.interp_degree == 1) return 3;  // linear
    if (p.interp_degree == 2) return 4;  // quadratic
    if (p.interp_degree == 3) return 5;  // cubic
    return 0;
  }

  if (p.interp_degree == 1 &&
      p.analy_degree == 0 &&
      p.synthe_degree == 1) {
    return 4;  // linear-antialiasing: total degree 2
  }

  if (p.interp_degree == 2 &&
      p.analy_degree == 1 &&
      p.synthe_degree == 2) {
    return 6;  // quadratic-antialiasing: total degree 4
  }

  if (p.interp_degree == 3 &&
      p.analy_degree == 1 &&
      p.synthe_degree == 3) {
    return 7;  // cubic-antialiasing: total degree 5
  }

  return 0;
}

static inline bool is_pure_quadratic_or_cubic_interp(const LSParams& p)
{
  return p.analy_degree < 0 &&
         p.synthe_degree == p.interp_degree &&
         (p.interp_degree == 2 || p.interp_degree == 3);
}

static inline bool is_linear_antialiasing_projection(const LSParams& p)
{
  return p.interp_degree == 1 &&
         p.analy_degree == 0 &&
         p.synthe_degree == 1;
}

static inline bool is_quadratic_antialiasing_projection(const LSParams& p)
{
  return p.interp_degree == 2 &&
         p.analy_degree == 1 &&
         p.synthe_degree == 2;
}

static inline bool is_cubic_antialiasing_projection(const LSParams& p)
{
  return p.interp_degree == 3 &&
         p.analy_degree == 1 &&
         p.synthe_degree == 3;
}

static inline bool is_oblique_antialiasing_projection(const LSParams& p)
{
  return is_linear_antialiasing_projection(p) ||
         is_quadratic_antialiasing_projection(p) ||
         is_cubic_antialiasing_projection(p);
}

static inline bool should_use_direct_3d_axis1_scatter(
  const std::vector<int64_t>& in_shape,
  const LSParams& p,
  const Plan1D& plan,
  int axis,
  int64_t nlines,
  int outN)
{
  constexpr int64_t kMinAxisOutputs = 1000000;
  if (!direct_3d_axis1_scatter_enabled() ||
      in_shape.size() != 3 ||
      axis != 1 ||
      !is_pure_quadratic_or_cubic_interp(p) ||
      nlines * static_cast<int64_t>(outN) < kMinAxisOutputs) {
    return false;
  }

  // Direct writes are helpful for single/small worker pools, but large pools
  // can make the extra destination traffic slower than the buffered row.
  return thread_count(
      nlines,
      plan,
      automatic_thread_cap_for_shape(in_shape)) <= 8;
}

static inline bool should_use_direct_2d_axis0_scatter(
  const std::vector<int64_t>& in_shape,
  const LSParams& p,
  int axis,
  int64_t nlines,
  int outN)
{
  constexpr int64_t kMinAxisOutputs = 250000;
  return direct_2d_axis0_scatter_enabled() &&
         in_shape.size() == 2 &&
         axis == 0 &&
         is_pure_quadratic_or_cubic_interp(p) &&
         p.zoom > 1.0 + 1e-12 &&
         nlines * static_cast<int64_t>(outN) >= kMinAxisOutputs;
}

static inline double interpolation_prefilter_lambda(int degree)
{
  if (degree <= 1) {
    return 1.0;
  }
  double lambda = 1.0;
  for (double z : spline_poles(degree)) {
    lambda *= (1.0 - z) * (1.0 - 1.0 / z);
  }
  return lambda;
}

static inline float interpolation_prefilter_lambda_f32(int degree)
{
  if (degree <= 1) {
    return 1.0f;
  }
  float lambda = 1.0f;
  for (double zd : spline_poles(degree)) {
    const float z = static_cast<float>(zd);
    lambda *= (1.0f - z) * (1.0f - 1.0f / z);
  }
  return lambda;
}

static inline int adaptive_batch_lines_for(
  const std::vector<int64_t>& in_shape,
  const LSParams& p,
  const Plan1D& plan,
  int64_t nlines,
  bool float32_internal)
{
  if (const int forced = env_positive_int_or_zero("LSRESIZE_BATCH_LINES")) {
    return forced;
  }

  // Row-major gather writes the coefficient block contiguously. Local profiles
  // show smaller 2-D pure downsampling batches keep that block hotter; other
  // methods/dimensions are noisier and keep the historical default.
  if (in_shape.size() == 2 &&
      is_pure_quadratic_or_cubic_interp(p) &&
      p.zoom < 1.0 - 1e-12) {
    if (batch_tune_v2_enabled()) {
      const int64_t max_dim = std::max(in_shape[0], in_shape[1]);
      const bool threaded = thread_count(nlines, plan) > 1;
      if (threaded) {
        return 16;
      }
      if (float32_internal) {
        return 24;
      }
      if (max_dim >= 2048) {
        return 24;
      }
      return 16;
    }
    return 16;
  }
  if (projection_batch_tune_enabled() &&
      in_shape.size() == 2 &&
      p.zoom < 1.0 - 1e-12) {
    if (is_linear_antialiasing_projection(p)) {
      return 32;
    }
    if (is_cubic_antialiasing_projection(p) &&
        std::max(in_shape[0], in_shape[1]) >= 1024) {
      const int64_t explicit_threads = explicit_thread_count(nlines);
      if (explicit_threads == 1) {
        return 32;
      }
      return 96;
    }
  }
  if (in_shape.size() == 3 &&
      float32_internal &&
      is_cubic_antialiasing_projection(p) &&
      p.zoom < 1.0 - 1e-12) {
    return 256;
  }

  return kDefaultBatchLines;
}

static inline bool should_use_batched_axis(
  BatchedAxisMode mode,
  const std::vector<int64_t>& in_shape,
  const LSParams& p,
  const Plan1D& plan,
  int64_t nlines)
{
  if (mode == BatchedAxisMode::Off) {
    return false;
  }
  if (mode == BatchedAxisMode::On) {
    return true;
  }

  if (p.interp_degree <= 0) {
    return false;
  }
  if (nlines < 128) {
    return false;
  }

  // Pure 3-D quadratic/cubic interpolation benefits from the batched
  // coefficient layout even for short axes: there are still many neighboring
  // lines, and batching avoids the per-line 1-D workspace pipeline.
  if (in_shape.size() == 3 &&
      is_pure_quadratic_or_cubic_interp(p)) {
    return true;
  }

  // 3-D oblique antialiasing has many neighboring short lines. The batched
  // projection kernel avoids per-line workspace overhead and is materially
  // faster on the PR benchmark cases, while 2-D projection remains governed by
  // the older size gates below.
  if (in_shape.size() == 3 &&
      is_oblique_antialiasing_projection(p)) {
    return true;
  }

  if (plan.N < 64 || plan.out_total < 64) {
    return false;
  }
  if (in_shape.size() == 2) {
    return true;
  }
  return false;
}

static inline void line_offsets(
  int64_t line,
  int axis,
  const std::vector<int>& bases,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  std::vector<int64_t>& idx,
  int64_t& in_off,
  int64_t& out_off)
{
  std::fill(idx.begin(), idx.end(), 0);

  int64_t t = line;
  for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
    const int d = bases[static_cast<size_t>(bi)];
    idx[static_cast<size_t>(d)] = t % in_shape[static_cast<size_t>(d)];
    t /= in_shape[static_cast<size_t>(d)];
  }

  in_off = 0;
  out_off = 0;
  for (int d = 0; d < static_cast<int>(idx.size()); ++d) {
    if (d != axis) {
      in_off  += idx[static_cast<size_t>(d)] *
                 in_strides[static_cast<size_t>(d)];
      out_off += idx[static_cast<size_t>(d)] *
                 out_strides[static_cast<size_t>(d)];
    }
  }
}

static inline void axis_pass_line_offsets(
  int64_t line,
  int axis,
  const std::vector<int>& bases,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  std::vector<int64_t>& idx,
  int64_t& in_off,
  int64_t& out_off)
{
  if (in_shape.size() == 2) {
    if (axis == 0) {
      in_off = line;
      out_off = line;
    } else {
      in_off = line * in_strides[0];
      out_off = line * out_strides[0];
    }
    return;
  }

  if (in_shape.size() == 3) {
    const int64_t s1 = in_shape[1];
    const int64_t s2 = in_shape[2];

    if (axis == 0) {
      const int64_t i2 = line % s2;
      const int64_t i1 = line / s2;
      in_off = i1 * in_strides[1] + i2 * in_strides[2];
      out_off = i1 * out_strides[1] + i2 * out_strides[2];
      return;
    }

    if (axis == 1) {
      const int64_t i2 = line % s2;
      const int64_t i0 = line / s2;
      in_off = i0 * in_strides[0] + i2 * in_strides[2];
      out_off = i0 * out_strides[0] + i2 * out_strides[2];
      return;
    }

    if (axis == 2) {
      const int64_t i1 = line % s1;
      const int64_t i0 = line / s1;
      in_off = i0 * in_strides[0] + i1 * in_strides[1];
      out_off = i0 * out_strides[0] + i1 * out_strides[1];
      return;
    }
  }

  line_offsets(
      line,
      axis,
      bases,
      in_shape,
      in_strides,
      out_strides,
      idx,
      in_off,
      out_off);
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_by_line(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  const int64_t* LS_RESTRICT in_offsets,
  int64_t axis_stride)
{
  for (int b = 0; b < B; ++b) {
    const InScalar* src = in + in_offsets[static_cast<size_t>(b)];
    CoeffScalar* dst = coeff + static_cast<size_t>(b);
    for (int n = 0; n < N; ++n) {
      *dst = static_cast<CoeffScalar>(
          src[static_cast<int64_t>(n) * axis_stride]);
      dst += Bs;
    }
  }
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_by_line_scaled(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  const int64_t* LS_RESTRICT in_offsets,
  int64_t axis_stride,
  CoeffScalar scale)
{
  for (int b = 0; b < B; ++b) {
    const InScalar* src = in + in_offsets[static_cast<size_t>(b)];
    CoeffScalar* dst = coeff + static_cast<size_t>(b);
    for (int n = 0; n < N; ++n) {
      *dst = scale * static_cast<CoeffScalar>(
          src[static_cast<int64_t>(n) * axis_stride]);
      dst += Bs;
    }
  }
}

static inline bool offsets_are_unit_stride(
  const int64_t* LS_RESTRICT offsets,
  int B)
{
  if (B <= 1) {
    return true;
  }
  const int64_t first = offsets[0];
  for (int b = 1; b < B; ++b) {
    if (offsets[static_cast<size_t>(b)] != first + b) {
      return false;
    }
  }
  return true;
}

static inline bool offsets_are_constant_stride(
  const int64_t* LS_RESTRICT offsets,
  int B,
  int64_t& stride)
{
  stride = 0;
  if (B <= 1) {
    return true;
  }
  stride = offsets[1] - offsets[0];
  for (int b = 2; b < B; ++b) {
    if (offsets[static_cast<size_t>(b)] != offsets[0] +
        static_cast<int64_t>(b) * stride) {
      return false;
    }
  }
  return true;
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_by_coeff_row(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  const int64_t* LS_RESTRICT in_offsets,
  int64_t axis_stride)
{
  for (int n = 0; n < N; ++n) {
    CoeffScalar* LS_RESTRICT dst = coeff + static_cast<size_t>(n) * Bs;
    const int64_t axis_delta = static_cast<int64_t>(n) * axis_stride;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = static_cast<CoeffScalar>(
          in[in_offsets[static_cast<size_t>(b)] + axis_delta]);
    }
  }
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_by_coeff_row_strided_offsets(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  int64_t first_offset,
  int64_t line_stride,
  int64_t axis_stride)
{
  for (int n = 0; n < N; ++n) {
    CoeffScalar* LS_RESTRICT dst = coeff + static_cast<size_t>(n) * Bs;
    const InScalar* LS_RESTRICT src =
        in + first_offset + static_cast<int64_t>(n) * axis_stride;
    if (line_stride == 1) {
      if constexpr (std::is_same_v<InScalar, CoeffScalar>) {
        std::memcpy(dst, src, static_cast<size_t>(B) * sizeof(CoeffScalar));
      } else {
        for (int b = 0; b < B; ++b) {
          dst[static_cast<size_t>(b)] =
              static_cast<CoeffScalar>(src[static_cast<size_t>(b)]);
        }
      }
    } else {
      for (int b = 0; b < B; ++b) {
        dst[static_cast<size_t>(b)] = static_cast<CoeffScalar>(
            src[static_cast<int64_t>(b) * line_stride]);
      }
    }
  }
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_by_coeff_row_scaled(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  const int64_t* LS_RESTRICT in_offsets,
  int64_t axis_stride,
  CoeffScalar scale)
{
  for (int n = 0; n < N; ++n) {
    CoeffScalar* LS_RESTRICT dst = coeff + static_cast<size_t>(n) * Bs;
    const int64_t axis_delta = static_cast<int64_t>(n) * axis_stride;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = scale * static_cast<CoeffScalar>(
          in[in_offsets[static_cast<size_t>(b)] + axis_delta]);
    }
  }
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_by_coeff_row_strided_offsets_scaled(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  int64_t first_offset,
  int64_t line_stride,
  int64_t axis_stride,
  CoeffScalar scale)
{
  for (int n = 0; n < N; ++n) {
    CoeffScalar* LS_RESTRICT dst = coeff + static_cast<size_t>(n) * Bs;
    const InScalar* LS_RESTRICT src =
        in + first_offset + static_cast<int64_t>(n) * axis_stride;
    if (line_stride == 1) {
      for (int b = 0; b < B; ++b) {
        dst[static_cast<size_t>(b)] =
            scale * static_cast<CoeffScalar>(src[static_cast<size_t>(b)]);
      }
    } else {
      for (int b = 0; b < B; ++b) {
        dst[static_cast<size_t>(b)] = scale * static_cast<CoeffScalar>(
            src[static_cast<int64_t>(b) * line_stride]);
      }
    }
  }
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  const int64_t* LS_RESTRICT in_offsets,
  int64_t axis_stride,
  bool row_major_gather,
  bool strided_offset_gather)
{
  if (row_major_gather) {
    int64_t line_stride = 0;
    if (strided_offset_gather &&
        offsets_are_constant_stride(in_offsets, B, line_stride)) {
      gather_axis_block_by_coeff_row_strided_offsets(
          in,
          coeff,
          Bs,
          B,
          N,
          in_offsets[0],
          line_stride,
          axis_stride);
      return;
    }
    gather_axis_block_by_coeff_row(
        in, coeff, Bs, B, N, in_offsets, axis_stride);
  } else {
    gather_axis_block_by_line(
        in, coeff, Bs, B, N, in_offsets, axis_stride);
  }
}

template <typename InScalar, typename CoeffScalar>
static inline void gather_axis_block_scaled(
  const InScalar* LS_RESTRICT in,
  CoeffScalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  int N,
  const int64_t* LS_RESTRICT in_offsets,
  int64_t axis_stride,
  bool row_major_gather,
  bool strided_offset_gather,
  CoeffScalar scale)
{
  if (row_major_gather) {
    int64_t line_stride = 0;
    if (strided_offset_gather &&
        offsets_are_constant_stride(in_offsets, B, line_stride)) {
      gather_axis_block_by_coeff_row_strided_offsets_scaled(
          in,
          coeff,
          Bs,
          B,
          N,
          in_offsets[0],
          line_stride,
          axis_stride,
          scale);
      return;
    }
    gather_axis_block_by_coeff_row_scaled(
        in, coeff, Bs, B, N, in_offsets, axis_stride, scale);
  } else {
    gather_axis_block_by_line_scaled(
        in, coeff, Bs, B, N, in_offsets, axis_stride, scale);
  }
}

static inline void accumulate_interior_row_colmajor(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  double* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int k0 = plan.kmin[static_cast<size_t>(l)];

  for (int t = begin; t < endw; ++t) {
    const double w = weights[static_cast<size_t>(t)];
    const int src = k0 + (t - begin);
    const double* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

static inline void accumulate_mapped_row_colmajor(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  double* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

  for (int t = begin; t < endw; ++t) {
    const size_t ti = static_cast<size_t>(t);
    const double w = weights[ti];
    const int src = coeff_src[ti];
    const double sgn = coeff_sgn[ti];
    const double* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] += w * (sgn * v[static_cast<size_t>(b)]);
    }
  }
}

static inline void accumulate_row_runs_colmajor(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  double* LS_RESTRICT y);

static inline void accumulate_interior_row_colmajor_f32_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  float* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int k0 = plan.kmin[static_cast<size_t>(l)];

  if (begin == endw) {
    std::fill(dst, dst + B, 0.0f);
    return;
  }

  {
    const float w = static_cast<float>(weights[static_cast<size_t>(begin)]);
    const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = w * v[static_cast<size_t>(b)];
    }
  }

  for (int t = begin + 1; t < endw; ++t) {
    const float w = static_cast<float>(weights[static_cast<size_t>(t)]);
    const int src = k0 + (t - begin);
    const float* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

static inline void accumulate_mapped_row_colmajor_f32_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  float* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

  if (begin == endw) {
    std::fill(dst, dst + B, 0.0f);
    return;
  }

  {
    const size_t ti = static_cast<size_t>(begin);
    const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
    const int src = coeff_src[ti];
    const float* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = w * v[static_cast<size_t>(b)];
    }
  }

  for (int t = begin + 1; t < endw; ++t) {
    const size_t ti = static_cast<size_t>(t);
    const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
    const int src = coeff_src[ti];
    const float* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

static inline void accumulate_row_colmajor_f32_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  bool interior,
  float* LS_RESTRICT dst)
{
  if (interior) {
    accumulate_interior_row_colmajor_f32_set(
        coeff, Bs, B, plan, weights, l, dst);
  } else {
    accumulate_mapped_row_colmajor_f32_set(
        coeff, Bs, B, plan, weights, l, dst);
  }
}

static inline void accumulate_row_runs_colmajor_f32_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  float* LS_RESTRICT y)
{
  const double* LS_RESTRICT weights = plan.weights.data();
  for (const RowRun1D& run : plan.row_runs) {
    for (int l = run.begin; l < run.end; ++l) {
      accumulate_row_colmajor_f32_set(
          coeff,
          Bs,
          B,
          plan,
          weights,
          l,
          run.interior != 0,
          y + static_cast<size_t>(l) * Bs);
    }
  }
}

template <int MaxM>
static inline void accumulate_interior_row_colmajor_f32_fixed_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  float* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int M = endw - begin;
  const int k0 = plan.kmin[static_cast<size_t>(l)];

  if (M <= 0) {
    std::fill(dst, dst + B, 0.0f);
    return;
  }
  if (M > MaxM) {
    accumulate_interior_row_colmajor_f32_set(
        coeff, Bs, B, plan, weights, l, dst);
    return;
  }

  const float w0 = static_cast<float>(weights[static_cast<size_t>(begin)]);
  const float* LS_RESTRICT v0 = coeff + static_cast<size_t>(k0) * Bs;
  for (int b = 0; b < B; ++b) {
    dst[static_cast<size_t>(b)] = w0 * v0[static_cast<size_t>(b)];
  }

  if constexpr (MaxM >= 2) {
    if (M >= 2) {
      const float w = static_cast<float>(weights[static_cast<size_t>(begin + 1)]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 1) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 3) {
    if (M >= 3) {
      const float w = static_cast<float>(weights[static_cast<size_t>(begin + 2)]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 2) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 4) {
    if (M >= 4) {
      const float w = static_cast<float>(weights[static_cast<size_t>(begin + 3)]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 3) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 5) {
    if (M >= 5) {
      const float w = static_cast<float>(weights[static_cast<size_t>(begin + 4)]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 4) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 6) {
    if (M >= 6) {
      const float w = static_cast<float>(weights[static_cast<size_t>(begin + 5)]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 5) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 7) {
    if (M >= 7) {
      const float w = static_cast<float>(weights[static_cast<size_t>(begin + 6)]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 6) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

template <int MaxM>
static inline void accumulate_mapped_row_colmajor_f32_fixed_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  float* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int M = endw - begin;
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

  if (M <= 0) {
    std::fill(dst, dst + B, 0.0f);
    return;
  }
  if (M > MaxM) {
    accumulate_mapped_row_colmajor_f32_set(
        coeff, Bs, B, plan, weights, l, dst);
    return;
  }

  const size_t t0 = static_cast<size_t>(begin);
  const float w0 = static_cast<float>(weights[t0] * coeff_sgn[t0]);
  const float* LS_RESTRICT v0 =
      coeff + static_cast<size_t>(coeff_src[t0]) * Bs;
  for (int b = 0; b < B; ++b) {
    dst[static_cast<size_t>(b)] = w0 * v0[static_cast<size_t>(b)];
  }

  if constexpr (MaxM >= 2) {
    if (M >= 2) {
      const size_t ti = static_cast<size_t>(begin + 1);
      const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 3) {
    if (M >= 3) {
      const size_t ti = static_cast<size_t>(begin + 2);
      const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 4) {
    if (M >= 4) {
      const size_t ti = static_cast<size_t>(begin + 3);
      const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 5) {
    if (M >= 5) {
      const size_t ti = static_cast<size_t>(begin + 4);
      const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 6) {
    if (M >= 6) {
      const size_t ti = static_cast<size_t>(begin + 5);
      const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 7) {
    if (M >= 7) {
      const size_t ti = static_cast<size_t>(begin + 6);
      const float w = static_cast<float>(weights[ti] * coeff_sgn[ti]);
      const float* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

template <int MaxM>
static inline void accumulate_row_colmajor_f32_fixed_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  bool interior,
  float* LS_RESTRICT dst)
{
  if (interior) {
    accumulate_interior_row_colmajor_f32_fixed_set<MaxM>(
        coeff, Bs, B, plan, weights, l, dst);
  } else {
    accumulate_mapped_row_colmajor_f32_fixed_set<MaxM>(
        coeff, Bs, B, plan, weights, l, dst);
  }
}

template <int MaxM>
static inline void accumulate_row_runs_colmajor_f32_fixed_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  float* LS_RESTRICT y)
{
  const double* LS_RESTRICT weights = plan.weights.data();
  for (const RowRun1D& run : plan.row_runs) {
    for (int l = run.begin; l < run.end; ++l) {
      accumulate_row_colmajor_f32_fixed_set<MaxM>(
          coeff,
          Bs,
          B,
          plan,
          weights,
          l,
          run.interior != 0,
          y + static_cast<size_t>(l) * Bs);
    }
  }
}

static inline void accumulate_row_runs_colmajor_f32_preset_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  int max_support,
  float* LS_RESTRICT y)
{
  switch (max_support) {
    case 3:
      accumulate_row_runs_colmajor_f32_fixed_set<3>(coeff, Bs, B, plan, y);
      return;
    case 4:
      accumulate_row_runs_colmajor_f32_fixed_set<4>(coeff, Bs, B, plan, y);
      return;
    case 5:
      accumulate_row_runs_colmajor_f32_fixed_set<5>(coeff, Bs, B, plan, y);
      return;
    case 6:
      accumulate_row_runs_colmajor_f32_fixed_set<6>(coeff, Bs, B, plan, y);
      return;
    case 7:
      accumulate_row_runs_colmajor_f32_fixed_set<7>(coeff, Bs, B, plan, y);
      return;
    default:
      break;
  }

  const size_t total =
      static_cast<size_t>(plan.out_total) * static_cast<size_t>(B);
  std::fill(y, y + total, 0.0f);
  accumulate_row_runs_colmajor_f32_set(coeff, Bs, B, plan, y);
}

static inline void accumulate_row_colmajor_f32_preset_set(
  const float* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int max_support,
  int l,
  bool interior,
  float* LS_RESTRICT dst)
{
  switch (max_support) {
    case 3:
      accumulate_row_colmajor_f32_fixed_set<3>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 4:
      accumulate_row_colmajor_f32_fixed_set<4>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 5:
      accumulate_row_colmajor_f32_fixed_set<5>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 6:
      accumulate_row_colmajor_f32_fixed_set<6>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 7:
      accumulate_row_colmajor_f32_fixed_set<7>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    default:
      break;
  }

  std::fill(dst, dst + B, 0.0f);
  accumulate_row_colmajor_f32_set(
      coeff, Bs, B, plan, weights, l, interior, dst);
}

static inline void accumulate_interior_row_colmajor_set_generic(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  double* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int k0 = plan.kmin[static_cast<size_t>(l)];

  if (begin == endw) {
    std::fill(dst, dst + B, 0.0);
    return;
  }

  {
    const double w = weights[static_cast<size_t>(begin)];
    const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = w * v[static_cast<size_t>(b)];
    }
  }

  for (int t = begin + 1; t < endw; ++t) {
    const double w = weights[static_cast<size_t>(t)];
    const int src = k0 + (t - begin);
    const double* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

static inline void accumulate_mapped_row_colmajor_set_generic(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  double* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

  if (begin == endw) {
    std::fill(dst, dst + B, 0.0);
    return;
  }

  {
    const size_t ti = static_cast<size_t>(begin);
    const double w = weights[ti] * coeff_sgn[ti];
    const int src = coeff_src[ti];
    const double* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = w * v[static_cast<size_t>(b)];
    }
  }

  for (int t = begin + 1; t < endw; ++t) {
    const size_t ti = static_cast<size_t>(t);
    const double w = weights[ti] * coeff_sgn[ti];
    const int src = coeff_src[ti];
    const double* LS_RESTRICT v = coeff + static_cast<size_t>(src) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

template <int MaxM>
static inline void accumulate_interior_row_colmajor_fixed_set(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  double* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int M = endw - begin;
  const int k0 = plan.kmin[static_cast<size_t>(l)];

  if (M <= 0) {
    std::fill(dst, dst + B, 0.0);
    return;
  }
  if (M > MaxM) {
    accumulate_interior_row_colmajor_set_generic(
        coeff, Bs, B, plan, weights, l, dst);
    return;
  }

  const double w0 = weights[static_cast<size_t>(begin)];
  const double* LS_RESTRICT v0 = coeff + static_cast<size_t>(k0) * Bs;
  for (int b = 0; b < B; ++b) {
    dst[static_cast<size_t>(b)] = w0 * v0[static_cast<size_t>(b)];
  }

  if constexpr (MaxM >= 2) {
    if (M >= 2) {
      const double w = weights[static_cast<size_t>(begin + 1)];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 1) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 3) {
    if (M >= 3) {
      const double w = weights[static_cast<size_t>(begin + 2)];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 2) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 4) {
    if (M >= 4) {
      const double w = weights[static_cast<size_t>(begin + 3)];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 3) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 5) {
    if (M >= 5) {
      const double w = weights[static_cast<size_t>(begin + 4)];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 4) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 6) {
    if (M >= 6) {
      const double w = weights[static_cast<size_t>(begin + 5)];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 5) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 7) {
    if (M >= 7) {
      const double w = weights[static_cast<size_t>(begin + 6)];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(k0 + 6) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

template <int MaxM>
static inline void accumulate_mapped_row_colmajor_fixed_set(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  double* LS_RESTRICT dst)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int M = endw - begin;
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

  if (M <= 0) {
    std::fill(dst, dst + B, 0.0);
    return;
  }
  if (M > MaxM) {
    accumulate_mapped_row_colmajor_set_generic(
        coeff, Bs, B, plan, weights, l, dst);
    return;
  }

  const size_t t0 = static_cast<size_t>(begin);
  const double w0 = weights[t0] * coeff_sgn[t0];
  const double* LS_RESTRICT v0 =
      coeff + static_cast<size_t>(coeff_src[t0]) * Bs;
  for (int b = 0; b < B; ++b) {
    dst[static_cast<size_t>(b)] = w0 * v0[static_cast<size_t>(b)];
  }

  if constexpr (MaxM >= 2) {
    if (M >= 2) {
      const size_t ti = static_cast<size_t>(begin + 1);
      const double w = weights[ti] * coeff_sgn[ti];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 3) {
    if (M >= 3) {
      const size_t ti = static_cast<size_t>(begin + 2);
      const double w = weights[ti] * coeff_sgn[ti];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 4) {
    if (M >= 4) {
      const size_t ti = static_cast<size_t>(begin + 3);
      const double w = weights[ti] * coeff_sgn[ti];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 5) {
    if (M >= 5) {
      const size_t ti = static_cast<size_t>(begin + 4);
      const double w = weights[ti] * coeff_sgn[ti];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 6) {
    if (M >= 6) {
      const size_t ti = static_cast<size_t>(begin + 5);
      const double w = weights[ti] * coeff_sgn[ti];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
  if constexpr (MaxM >= 7) {
    if (M >= 7) {
      const size_t ti = static_cast<size_t>(begin + 6);
      const double w = weights[ti] * coeff_sgn[ti];
      const double* LS_RESTRICT v = coeff + static_cast<size_t>(coeff_src[ti]) * Bs;
      for (int b = 0; b < B; ++b) dst[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
    }
  }
}

template <int MaxM>
static inline void accumulate_row_colmajor_fixed_set(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  bool interior,
  double* LS_RESTRICT dst)
{
  if (interior) {
    accumulate_interior_row_colmajor_fixed_set<MaxM>(
        coeff, Bs, B, plan, weights, l, dst);
  } else {
    accumulate_mapped_row_colmajor_fixed_set<MaxM>(
        coeff, Bs, B, plan, weights, l, dst);
  }
}

template <int MaxM>
static inline void accumulate_row_runs_colmajor_fixed_set(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  double* LS_RESTRICT y)
{
  const double* LS_RESTRICT weights = plan.weights.data();
  for (const RowRun1D& run : plan.row_runs) {
    for (int l = run.begin; l < run.end; ++l) {
      accumulate_row_colmajor_fixed_set<MaxM>(
          coeff,
          Bs,
          B,
          plan,
          weights,
          l,
          run.interior != 0,
          y + static_cast<size_t>(l) * Bs);
    }
  }
}

static inline void accumulate_row_runs_colmajor_preset_set(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  int max_support,
  double* LS_RESTRICT y)
{
  switch (max_support) {
    case 3:
      accumulate_row_runs_colmajor_fixed_set<3>(coeff, Bs, B, plan, y);
      return;
    case 4:
      accumulate_row_runs_colmajor_fixed_set<4>(coeff, Bs, B, plan, y);
      return;
    case 5:
      accumulate_row_runs_colmajor_fixed_set<5>(coeff, Bs, B, plan, y);
      return;
    case 6:
      accumulate_row_runs_colmajor_fixed_set<6>(coeff, Bs, B, plan, y);
      return;
    case 7:
      accumulate_row_runs_colmajor_fixed_set<7>(coeff, Bs, B, plan, y);
      return;
    default:
      break;
  }

  const size_t total =
      static_cast<size_t>(plan.out_total) * static_cast<size_t>(B);
  std::fill(y, y + total, 0.0);
  accumulate_row_runs_colmajor(coeff, Bs, B, plan, y);
}

static inline void accumulate_row_colmajor_preset_set(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int max_support,
  int l,
  bool interior,
  double* LS_RESTRICT dst)
{
  switch (max_support) {
    case 3:
      accumulate_row_colmajor_fixed_set<3>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 4:
      accumulate_row_colmajor_fixed_set<4>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 5:
      accumulate_row_colmajor_fixed_set<5>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 6:
      accumulate_row_colmajor_fixed_set<6>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    case 7:
      accumulate_row_colmajor_fixed_set<7>(
          coeff, Bs, B, plan, weights, l, interior, dst);
      return;
    default:
      break;
  }

  std::fill(dst, dst + B, 0.0);
  if (interior) {
    accumulate_interior_row_colmajor(
        coeff, Bs, B, plan, weights, l, dst);
  } else {
    accumulate_mapped_row_colmajor(
        coeff, Bs, B, plan, weights, l, dst);
  }
}

template <typename Scalar>
static inline void scatter_row_zero(
  int B,
  Scalar* LS_RESTRICT out,
  const int64_t* LS_RESTRICT out_offsets,
  int64_t out_delta,
  bool unit_stride_offsets)
{
  if (unit_stride_offsets) {
    Scalar* LS_RESTRICT dst = out + out_offsets[0] + out_delta;
    std::fill(dst, dst + B, Scalar(0));
    return;
  }

  for (int b = 0; b < B; ++b) {
    out[out_offsets[static_cast<size_t>(b)] + out_delta] = Scalar(0);
  }
}

template <typename Scalar>
static inline void scatter_row_assign_scaled(
  const Scalar* LS_RESTRICT values,
  Scalar weight,
  int B,
  Scalar* LS_RESTRICT out,
  const int64_t* LS_RESTRICT out_offsets,
  int64_t out_delta,
  bool unit_stride_offsets)
{
  if (unit_stride_offsets) {
    Scalar* LS_RESTRICT dst = out + out_offsets[0] + out_delta;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = weight * values[static_cast<size_t>(b)];
    }
    return;
  }

  for (int b = 0; b < B; ++b) {
    out[out_offsets[static_cast<size_t>(b)] + out_delta] =
        weight * values[static_cast<size_t>(b)];
  }
}

template <typename Scalar>
static inline void scatter_row_add_scaled(
  const Scalar* LS_RESTRICT values,
  Scalar weight,
  int B,
  Scalar* LS_RESTRICT out,
  const int64_t* LS_RESTRICT out_offsets,
  int64_t out_delta,
  bool unit_stride_offsets)
{
  if (unit_stride_offsets) {
    Scalar* LS_RESTRICT dst = out + out_offsets[0] + out_delta;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] +=
          weight * values[static_cast<size_t>(b)];
    }
    return;
  }

  for (int b = 0; b < B; ++b) {
    out[out_offsets[static_cast<size_t>(b)] + out_delta] +=
        weight * values[static_cast<size_t>(b)];
  }
}

template <typename Scalar, bool Mapped, int MaxM>
static inline void accumulate_scatter_row_colmajor_impl(
  const Scalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  Scalar* LS_RESTRICT out,
  const int64_t* LS_RESTRICT out_offsets,
  int64_t stride,
  bool unit_stride_offsets)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  const int M = endw - begin;
  const int k0 = plan.kmin[static_cast<size_t>(l)];
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();
  const int64_t out_delta = static_cast<int64_t>(l) * stride;

  if (M <= 0) {
    scatter_row_zero(B, out, out_offsets, out_delta, unit_stride_offsets);
    return;
  }
  if constexpr (MaxM > 0) {
    if (M > MaxM) {
      accumulate_scatter_row_colmajor_impl<Scalar, Mapped, 0>(
          coeff, Bs, B, plan, weights, l, out, out_offsets, stride,
          unit_stride_offsets);
      return;
    }
  }

  auto row_values = [&](int t) -> const Scalar* {
    if constexpr (Mapped) {
      return coeff + static_cast<size_t>(coeff_src[static_cast<size_t>(t)]) * Bs;
    } else {
      return coeff + static_cast<size_t>(k0 + (t - begin)) * Bs;
    }
  };
  auto row_weight = [&](int t) -> Scalar {
    const size_t ti = static_cast<size_t>(t);
    if constexpr (Mapped) {
      return static_cast<Scalar>(weights[ti] * coeff_sgn[ti]);
    } else {
      return static_cast<Scalar>(weights[ti]);
    }
  };

  scatter_row_assign_scaled(
      row_values(begin),
      row_weight(begin),
      B,
      out,
      out_offsets,
      out_delta,
      unit_stride_offsets);

  if constexpr (MaxM == 0) {
    for (int t = begin + 1; t < endw; ++t) {
      scatter_row_add_scaled(
          row_values(t),
          row_weight(t),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
    return;
  }

  if constexpr (MaxM >= 2) {
    if (M >= 2) {
      scatter_row_add_scaled(
          row_values(begin + 1),
          row_weight(begin + 1),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
  }
  if constexpr (MaxM >= 3) {
    if (M >= 3) {
      scatter_row_add_scaled(
          row_values(begin + 2),
          row_weight(begin + 2),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
  }
  if constexpr (MaxM >= 4) {
    if (M >= 4) {
      scatter_row_add_scaled(
          row_values(begin + 3),
          row_weight(begin + 3),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
  }
  if constexpr (MaxM >= 5) {
    if (M >= 5) {
      scatter_row_add_scaled(
          row_values(begin + 4),
          row_weight(begin + 4),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
  }
  if constexpr (MaxM >= 6) {
    if (M >= 6) {
      scatter_row_add_scaled(
          row_values(begin + 5),
          row_weight(begin + 5),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
  }
  if constexpr (MaxM >= 7) {
    if (M >= 7) {
      scatter_row_add_scaled(
          row_values(begin + 6),
          row_weight(begin + 6),
          B,
          out,
          out_offsets,
          out_delta,
          unit_stride_offsets);
    }
  }
}

template <typename Scalar, int MaxM>
static inline void accumulate_scatter_row_colmajor_fixed_set(
  const Scalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  bool interior,
  Scalar* LS_RESTRICT out,
  const int64_t* LS_RESTRICT out_offsets,
  int64_t stride,
  bool unit_stride_offsets)
{
  if (interior) {
    accumulate_scatter_row_colmajor_impl<Scalar, false, MaxM>(
        coeff, Bs, B, plan, weights, l, out, out_offsets, stride,
        unit_stride_offsets);
  } else {
    accumulate_scatter_row_colmajor_impl<Scalar, true, MaxM>(
        coeff, Bs, B, plan, weights, l, out, out_offsets, stride,
        unit_stride_offsets);
  }
}

template <typename Scalar>
static inline void accumulate_scatter_row_colmajor_preset_set(
  const Scalar* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int max_support,
  int l,
  bool interior,
  Scalar* LS_RESTRICT out,
  const int64_t* LS_RESTRICT out_offsets,
  int64_t stride,
  bool unit_stride_offsets)
{
  switch (max_support) {
    case 3:
      accumulate_scatter_row_colmajor_fixed_set<Scalar, 3>(
          coeff, Bs, B, plan, weights, l, interior, out, out_offsets, stride,
          unit_stride_offsets);
      return;
    case 4:
      accumulate_scatter_row_colmajor_fixed_set<Scalar, 4>(
          coeff, Bs, B, plan, weights, l, interior, out, out_offsets, stride,
          unit_stride_offsets);
      return;
    case 5:
      accumulate_scatter_row_colmajor_fixed_set<Scalar, 5>(
          coeff, Bs, B, plan, weights, l, interior, out, out_offsets, stride,
          unit_stride_offsets);
      return;
    case 6:
      accumulate_scatter_row_colmajor_fixed_set<Scalar, 6>(
          coeff, Bs, B, plan, weights, l, interior, out, out_offsets, stride,
          unit_stride_offsets);
      return;
    case 7:
      accumulate_scatter_row_colmajor_fixed_set<Scalar, 7>(
          coeff, Bs, B, plan, weights, l, interior, out, out_offsets, stride,
          unit_stride_offsets);
      return;
    default:
      break;
  }

  if (interior) {
    accumulate_scatter_row_colmajor_impl<Scalar, false, 0>(
        coeff, Bs, B, plan, weights, l, out, out_offsets, stride,
        unit_stride_offsets);
  } else {
    accumulate_scatter_row_colmajor_impl<Scalar, true, 0>(
        coeff, Bs, B, plan, weights, l, out, out_offsets, stride,
        unit_stride_offsets);
  }
}

static inline void accumulate_row_runs_colmajor(
  const double* LS_RESTRICT coeff,
  size_t Bs,
  int B,
  const Plan1D& plan,
  double* LS_RESTRICT y)
{
  const double* LS_RESTRICT weights = plan.weights.data();
  for (const RowRun1D& run : plan.row_runs) {
    if (run.interior) {
      for (int l = run.begin; l < run.end; ++l) {
        accumulate_interior_row_colmajor(
            coeff, Bs, B, plan, weights, l, y + static_cast<size_t>(l) * Bs);
      }
    } else {
      for (int l = run.begin; l < run.end; ++l) {
        accumulate_mapped_row_colmajor(
            coeff, Bs, B, plan, weights, l, y + static_cast<size_t>(l) * Bs);
      }
    }
  }
}

template <typename Scalar>
static inline double accumulate_scalar_line_plan_row(
  const Scalar* LS_RESTRICT line,
  int64_t stride,
  const Plan1D& plan,
  const double* LS_RESTRICT weights,
  int l,
  bool interior)
{
  const int begin = plan.row_ptr[static_cast<size_t>(l)];
  const int endw = plan.row_ptr[static_cast<size_t>(l) + 1];
  double acc = 0.0;

  if (interior) {
    const int k0 = plan.kmin[static_cast<size_t>(l)];
    for (int t = begin; t < endw; ++t) {
      const int src = k0 + (t - begin);
      acc += weights[static_cast<size_t>(t)] *
             static_cast<double>(line[static_cast<int64_t>(src) * stride]);
    }
    return acc;
  }

  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const std::int8_t* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();
  for (int t = begin; t < endw; ++t) {
    const size_t ti = static_cast<size_t>(t);
    const int src = coeff_src[ti];
    acc += weights[ti] * coeff_sgn[ti] *
           static_cast<double>(line[static_cast<int64_t>(src) * stride]);
  }
  return acc;
}

template <typename Weight>
struct DirectLinearPlanView {
  bool ok = false;
  const unsigned char* LS_RESTRICT count = nullptr;
  const int* LS_RESTRICT src0 = nullptr;
  const int* LS_RESTRICT src1 = nullptr;
  const int* LS_RESTRICT src2 = nullptr;
  const Weight* LS_RESTRICT w0 = nullptr;
  const Weight* LS_RESTRICT w1 = nullptr;
  const Weight* LS_RESTRICT w2 = nullptr;
};

template <typename Weight>
static inline DirectLinearPlanView<Weight> direct_linear_plan_view(
  const Plan1D& plan)
{
  DirectLinearPlanView<Weight> view;
  if (!plan.direct_linear_ok) {
    return view;
  }

  view.ok = true;
  view.count = plan.direct_linear_count.data();
  view.src0 = plan.direct_linear_src0.data();
  view.src1 = plan.direct_linear_src1.data();
  view.src2 = plan.direct_linear_src2.data();
  if constexpr (std::is_same_v<Weight, float>) {
    view.w0 = plan.direct_linear_w0_f32.data();
    view.w1 = plan.direct_linear_w1_f32.data();
    view.w2 = plan.direct_linear_w2_f32.data();
  } else {
    view.w0 = plan.direct_linear_w0.data();
    view.w1 = plan.direct_linear_w1.data();
    view.w2 = plan.direct_linear_w2.data();
  }
  return view;
}

static inline std::pair<int64_t, int64_t> longest_count_run(
  const unsigned char* LS_RESTRICT count,
  int64_t n,
  unsigned char target)
{
  int64_t best_begin = 0;
  int64_t best_end = 0;
  int64_t run_begin = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (count[static_cast<size_t>(i)] == target) {
      if (run_begin < 0) {
        run_begin = i;
      }
      continue;
    }
    if (run_begin >= 0 && i - run_begin > best_end - best_begin) {
      best_begin = run_begin;
      best_end = i;
    }
    run_begin = -1;
  }
  if (run_begin >= 0 && n - run_begin > best_end - best_begin) {
    best_begin = run_begin;
    best_end = n;
  }
  return {best_begin, best_end};
}

static inline std::pair<int64_t, int64_t> longest_count_run(
  const std::vector<unsigned char>& count,
  unsigned char target)
{
  return longest_count_run(
      count.data(),
      static_cast<int64_t>(count.size()),
      target);
}

#if LSRESIZE_GNU_X86_TARGETS
__attribute__((target("avx2,fma")))
static void resize_linear_axis1_col2_f32_avx2(
  const float* LS_RESTRICT src,
  float* LS_RESTRICT dst,
  int64_t begin,
  int64_t end,
  const int* LS_RESTRICT s0,
  const int* LS_RESTRICT s1,
  const float* LS_RESTRICT w0,
  const float* LS_RESTRICT w1)
{
  int64_t l = begin;
  for (; l + 8 <= end; l += 8) {
    const size_t li = static_cast<size_t>(l);
    const __m256i idx0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s0 + li));
    const __m256i idx1 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s1 + li));
    const __m256 w0v = _mm256_loadu_ps(w0 + li);
    const __m256 w1v = _mm256_loadu_ps(w1 + li);
    const __m256 v0 = _mm256_i32gather_ps(src, idx0, 4);
    const __m256 v1 = _mm256_i32gather_ps(src, idx1, 4);
    const __m256 acc = _mm256_fmadd_ps(w1v, v1, _mm256_mul_ps(w0v, v0));
    _mm256_storeu_ps(dst + li, acc);
  }

  for (; l < end; ++l) {
    const size_t li = static_cast<size_t>(l);
    dst[li] = w0[li] * src[static_cast<size_t>(s0[li])] +
              w1[li] * src[static_cast<size_t>(s1[li])];
  }
}

__attribute__((target("avx2,fma")))
static void resize_linear_axis1_col2_f64_avx2(
  const double* LS_RESTRICT src,
  double* LS_RESTRICT dst,
  int64_t begin,
  int64_t end,
  const int* LS_RESTRICT s0,
  const int* LS_RESTRICT s1,
  const double* LS_RESTRICT w0,
  const double* LS_RESTRICT w1)
{
  int64_t l = begin;
  for (; l + 4 <= end; l += 4) {
    const size_t li = static_cast<size_t>(l);
    const __m128i idx0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0 + li));
    const __m128i idx1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(s1 + li));
    const __m256d w0v = _mm256_loadu_pd(w0 + li);
    const __m256d w1v = _mm256_loadu_pd(w1 + li);
    const __m256d v0 = _mm256_i32gather_pd(src, idx0, 8);
    const __m256d v1 = _mm256_i32gather_pd(src, idx1, 8);
    const __m256d acc = _mm256_fmadd_pd(w1v, v1, _mm256_mul_pd(w0v, v0));
    _mm256_storeu_pd(dst + li, acc);
  }

  for (; l < end; ++l) {
    const size_t li = static_cast<size_t>(l);
    dst[li] = w0[li] * src[static_cast<size_t>(s0[li])] +
              w1[li] * src[static_cast<size_t>(s1[li])];
  }
}
#endif

template <typename Scalar>
static void resize_along_axis_2d_linear_direct(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const Plan1D& plan,
  int64_t nlines)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::LinearDirectTotal);

  const int64_t in_w = in_shape[1];
  const int64_t out_w = out_shape[1];
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;
  const auto direct = direct_linear_plan_view<Accum>(plan);

  if (!direct.ok) {
    const double* LS_RESTRICT weights = plan.weights.data();
    auto fallback = [&](int64_t start, int64_t end) {
      if (axis == 1) {
        for (int64_t row = start; row < end; ++row) {
          const Scalar* LS_RESTRICT src = in + row * in_w;
          Scalar* LS_RESTRICT dst = out + row * out_w;
          for (const RowRun1D& run : plan.row_runs) {
            const bool interior = (run.interior != 0);
            for (int l = run.begin; l < run.end; ++l) {
              dst[static_cast<size_t>(l)] = static_cast<Scalar>(
                  accumulate_scalar_line_plan_row(
                      src,
                      1,
                      plan,
                      weights,
                      l,
                      interior));
            }
          }
        }
        return;
      }

      for (const RowRun1D& run : plan.row_runs) {
        const bool interior = (run.interior != 0);
        for (int l = run.begin; l < run.end; ++l) {
          Scalar* LS_RESTRICT dst =
              out + static_cast<int64_t>(l) * out_w + start;
          for (int64_t col = start; col < end; ++col) {
            dst[static_cast<size_t>(col - start)] = static_cast<Scalar>(
                accumulate_scalar_line_plan_row(
                    in + col,
                    in_w,
                    plan,
                    weights,
                    l,
                    interior));
          }
        }
      }
    };

    run_parallel_for_shape(nlines, plan, in_shape, fallback);
    return;
  }

  const unsigned char* LS_RESTRICT c = direct.count;
  const int* LS_RESTRICT s0 = direct.src0;
  const int* LS_RESTRICT s1 = direct.src1;
  const int* LS_RESTRICT s2 = direct.src2;
  const Accum* LS_RESTRICT a0 = direct.w0;
  const Accum* LS_RESTRICT a1 = direct.w1;
  const Accum* LS_RESTRICT a2 = direct.w2;
  const auto count2_run = longest_count_run(c, plan.outN, 2);
  const int64_t count2_begin = count2_run.first;
  const int64_t count2_end = count2_run.second;
  const bool use_avx2_axis1 =
      axis == 1 &&
      avx2_linear_enabled() &&
      (count2_end - count2_begin >= 16);

  auto worker = [&](int64_t start, int64_t end) {
    if (axis == 1) {
      for (int64_t row = start; row < end; ++row) {
        const Scalar* LS_RESTRICT src = in + row * in_w;
        Scalar* LS_RESTRICT dst = out + row * out_w;
        auto scalar_axis1 = [&](int64_t l_begin, int64_t l_end) {
          for (int64_t l = l_begin; l < l_end; ++l) {
            const size_t li = static_cast<size_t>(l);
            Accum acc = Accum(0);
            switch (c[li]) {
              case 3:
                acc = a0[li] * static_cast<Accum>(src[static_cast<size_t>(s0[li])]);
                acc += a1[li] * static_cast<Accum>(src[static_cast<size_t>(s1[li])]);
                acc += a2[li] * static_cast<Accum>(src[static_cast<size_t>(s2[li])]);
                break;
              case 2:
                acc = a0[li] * static_cast<Accum>(src[static_cast<size_t>(s0[li])]);
                acc += a1[li] * static_cast<Accum>(src[static_cast<size_t>(s1[li])]);
                break;
              case 1:
                acc = a0[li] * static_cast<Accum>(src[static_cast<size_t>(s0[li])]);
                break;
              default:
                break;
            }
            dst[li] = static_cast<Scalar>(acc);
          }
        };

#if LSRESIZE_GNU_X86_TARGETS
        if (use_avx2_axis1) {
          scalar_axis1(0, count2_begin);
          if constexpr (std::is_same_v<Scalar, float>) {
            resize_linear_axis1_col2_f32_avx2(
                src, dst, count2_begin, count2_end, s0, s1, a0, a1);
          } else if constexpr (std::is_same_v<Scalar, double>) {
            resize_linear_axis1_col2_f64_avx2(
                src, dst, count2_begin, count2_end, s0, s1, a0, a1);
          }
          scalar_axis1(count2_end, plan.outN);
          continue;
        }
#endif

        scalar_axis1(0, plan.outN);
      }
      return;
    }

    for (int l = 0; l < plan.outN; ++l) {
      const size_t li = static_cast<size_t>(l);
      Scalar* LS_RESTRICT dst = out + static_cast<int64_t>(l) * out_w + start;
      const Scalar* LS_RESTRICT row0 =
          in + static_cast<int64_t>(s0[li]) * in_w + start;
      const Scalar* LS_RESTRICT row1 =
          in + static_cast<int64_t>(s1[li]) * in_w + start;
      const Scalar* LS_RESTRICT row2 =
          in + static_cast<int64_t>(s2[li]) * in_w + start;
      switch (c[li]) {
        case 3:
          for (int64_t col = start; col < end; ++col) {
            const size_t ci = static_cast<size_t>(col - start);
            Accum acc = a0[li] * static_cast<Accum>(row0[ci]);
            acc += a1[li] * static_cast<Accum>(row1[ci]);
            acc += a2[li] * static_cast<Accum>(row2[ci]);
            dst[ci] = static_cast<Scalar>(acc);
          }
          break;
        case 2:
          for (int64_t col = start; col < end; ++col) {
            const size_t ci = static_cast<size_t>(col - start);
            Accum acc = a0[li] * static_cast<Accum>(row0[ci]);
            acc += a1[li] * static_cast<Accum>(row1[ci]);
            dst[ci] = static_cast<Scalar>(acc);
          }
          break;
        case 1:
          for (int64_t col = start; col < end; ++col) {
            const size_t ci = static_cast<size_t>(col - start);
            dst[ci] = static_cast<Scalar>(
                a0[li] * static_cast<Accum>(row0[ci]));
          }
          break;
        default:
          for (int64_t col = start; col < end; ++col) {
            dst[static_cast<size_t>(col - start)] = static_cast<Scalar>(0);
          }
          break;
        }
    }
  };

  run_parallel_for_shape(nlines, plan, in_shape, worker);
}

template <typename Scalar>
static void resize_along_axis_linear_direct(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  int axis,
  const Plan1D& plan,
  const std::vector<int>& bases,
  int64_t nlines)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::LinearDirectTotal);

  const int D = static_cast<int>(in_shape.size());
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;
  const auto direct = direct_linear_plan_view<Accum>(plan);

  if (!direct.ok) {
    const double* LS_RESTRICT weights = plan.weights.data();
    auto fallback = [&](int64_t start, int64_t end) {
      std::vector<int64_t> idx(static_cast<size_t>(D), 0);
      for (int64_t line = start; line < end; ++line) {
        int64_t in_off = 0;
        int64_t out_off = 0;
        line_offsets(
            line,
            axis,
            bases,
            in_shape,
            in_strides,
            out_strides,
            idx,
            in_off,
            out_off);
        const Scalar* LS_RESTRICT src = in + in_off;
        const int64_t in_stride = in_strides[static_cast<size_t>(axis)];
        const int64_t out_stride = out_strides[static_cast<size_t>(axis)];
        Scalar* LS_RESTRICT dst = out + out_off;
        for (const RowRun1D& run : plan.row_runs) {
          const bool interior = (run.interior != 0);
          for (int l = run.begin; l < run.end; ++l) {
            dst[static_cast<int64_t>(l) * out_stride] = static_cast<Scalar>(
                accumulate_scalar_line_plan_row(
                    src,
                    in_stride,
                    plan,
                    weights,
                    l,
                    interior));
          }
        }
      }
    };

    run_parallel_for_shape(nlines, plan, in_shape, fallback);
    return;
  }

  const unsigned char* LS_RESTRICT c = direct.count;
  const int* LS_RESTRICT s0 = direct.src0;
  const int* LS_RESTRICT s1 = direct.src1;
  const int* LS_RESTRICT s2 = direct.src2;
  const Accum* LS_RESTRICT a0 = direct.w0;
  const Accum* LS_RESTRICT a1 = direct.w1;
  const Accum* LS_RESTRICT a2 = direct.w2;

  if (axis == D - 1 && last_axis_linear_direct_enabled()) {
    const auto count2_run = longest_count_run(c, plan.outN, 2);
    const int64_t count2_begin = count2_run.first;
    const int64_t count2_end = count2_run.second;
    const bool use_avx2_last_axis =
        avx2_linear_enabled() &&
        (count2_end - count2_begin >= 16);

    auto worker = [&](int64_t start, int64_t end) {
      for (int64_t line = start; line < end; ++line) {
        const Scalar* LS_RESTRICT src = in + line * static_cast<int64_t>(plan.N);
        Scalar* LS_RESTRICT dst = out + line * static_cast<int64_t>(plan.outN);
        auto scalar_last_axis = [&](int64_t l_begin, int64_t l_end) {
          for (int64_t l = l_begin; l < l_end; ++l) {
            const size_t li = static_cast<size_t>(l);
            Accum acc = Accum(0);
            switch (c[li]) {
              case 3:
                acc = a0[li] *
                      static_cast<Accum>(src[static_cast<size_t>(s0[li])]);
                acc += a1[li] *
                       static_cast<Accum>(src[static_cast<size_t>(s1[li])]);
                acc += a2[li] *
                       static_cast<Accum>(src[static_cast<size_t>(s2[li])]);
                break;
              case 2:
                acc = a0[li] *
                      static_cast<Accum>(src[static_cast<size_t>(s0[li])]);
                acc += a1[li] *
                       static_cast<Accum>(src[static_cast<size_t>(s1[li])]);
                break;
              case 1:
                acc = a0[li] *
                      static_cast<Accum>(src[static_cast<size_t>(s0[li])]);
                break;
              default:
                break;
            }
            dst[li] = static_cast<Scalar>(acc);
          }
        };

#if LSRESIZE_GNU_X86_TARGETS
        if (use_avx2_last_axis) {
          scalar_last_axis(0, count2_begin);
          if constexpr (std::is_same_v<Scalar, float>) {
            resize_linear_axis1_col2_f32_avx2(
                src, dst, count2_begin, count2_end, s0, s1, a0, a1);
          } else if constexpr (std::is_same_v<Scalar, double>) {
            resize_linear_axis1_col2_f64_avx2(
                src, dst, count2_begin, count2_end, s0, s1, a0, a1);
          }
          scalar_last_axis(count2_end, plan.outN);
          continue;
        }
#endif

        scalar_last_axis(0, plan.outN);
      }
    };

    run_parallel_for_shape(nlines, plan, in_shape, worker);
    return;
  }

  auto worker = [&](int64_t start, int64_t end) {
    std::vector<int64_t> idx(static_cast<size_t>(D), 0);
    for (int64_t line = start; line < end; ++line) {
      int64_t in_off = 0;
      int64_t out_off = 0;
      line_offsets(
          line,
          axis,
          bases,
          in_shape,
          in_strides,
          out_strides,
          idx,
          in_off,
          out_off);
      const Scalar* LS_RESTRICT src = in + in_off;
      Scalar* LS_RESTRICT dst = out + out_off;
      const int64_t in_stride = in_strides[static_cast<size_t>(axis)];
      const int64_t out_stride = out_strides[static_cast<size_t>(axis)];

      for (int l = 0; l < plan.outN; ++l) {
        const size_t li = static_cast<size_t>(l);
        Accum acc = Accum(0);
        switch (c[li]) {
          case 3:
            acc = a0[li] *
                  static_cast<Accum>(src[static_cast<int64_t>(s0[li]) * in_stride]);
            acc += a1[li] *
                   static_cast<Accum>(src[static_cast<int64_t>(s1[li]) * in_stride]);
            acc += a2[li] *
                   static_cast<Accum>(src[static_cast<int64_t>(s2[li]) * in_stride]);
            break;
          case 2:
            acc = a0[li] *
                  static_cast<Accum>(src[static_cast<int64_t>(s0[li]) * in_stride]);
            acc += a1[li] *
                   static_cast<Accum>(src[static_cast<int64_t>(s1[li]) * in_stride]);
            break;
          case 1:
            acc = a0[li] *
                  static_cast<Accum>(src[static_cast<int64_t>(s0[li]) * in_stride]);
            break;
          default:
            break;
        }
        dst[static_cast<int64_t>(l) * out_stride] = static_cast<Scalar>(acc);
      }
    }
  };

  run_parallel_for_shape(nlines, plan, in_shape, worker);
}

static inline bool can_use_linear_interp_fast_path(
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p)
{
  return fast_linear_interp_enabled() &&
         !in_shape.empty() &&
         in_shape.size() == out_shape.size() &&
         axis >= 0 &&
         axis < static_cast<int>(in_shape.size()) &&
         p.analy_degree < 0 &&
         p.synthe_degree == p.interp_degree &&
         p.interp_degree == 1;
}

template <typename Scalar>
static void resize_along_axis_2d_linear_fast(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const Plan1D& plan,
  int64_t nlines)
{
  resize_along_axis_2d_linear_direct(
      in,
      out,
      in_shape,
      out_shape,
      axis,
      plan,
      nlines);
}

template <typename Scalar>
static void resize_along_axis_batched_interp_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  int axis,
  const LSParams& p,
  const Plan1D& plan,
  const std::vector<int>& bases,
  int64_t nlines)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpTotal);

  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int batch_lines =
      std::max(1, adaptive_batch_lines_for(
          in_shape, p, plan, nlines, false));
  const int64_t axis_stride_in = in_strides[static_cast<size_t>(axis)];
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const bool row_major_gather = row_gather_enabled();
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p, plan)
                        : 0;
  const bool direct_axis1_scatter =
      should_use_direct_3d_axis1_scatter(in_shape, p, plan, axis, nlines, outN);
  const bool direct_2d_axis0_scatter =
      should_use_direct_2d_axis0_scatter(in_shape, p, axis, nlines, outN);
  const bool scaled_gather_prefilter =
      gather_prefilter_scale_enabled() && p.interp_degree > 1 && N > 1;
  const double gather_scale = scaled_gather_prefilter
                            ? interpolation_prefilter_lambda(p.interp_degree)
                            : 1.0;
  const int64_t max_input_dim =
      *std::max_element(in_shape.begin(), in_shape.end());
  const bool strided_offset_gather_candidate =
      p.zoom <= 1.0 &&
      (in_shape.size() == 2 ? max_input_dim >= 1024
                            : max_input_dim >= 192);
  const bool strided_offset_gather =
      strided_offset_gather_candidate && strided_offset_gather_enabled();

  auto worker = [&](int64_t start, int64_t end) {
    std::vector<int64_t> idx(D, 0);
    std::vector<int64_t> in_offsets(static_cast<size_t>(batch_lines));
    std::vector<int64_t> out_offsets(static_cast<size_t>(batch_lines));
    std::vector<double> coeff;
    std::vector<double> y;
    std::vector<double> accum;
    coeff.reserve(static_cast<size_t>(N) * static_cast<size_t>(batch_lines));
    y.reserve(static_cast<size_t>(outN) * static_cast<size_t>(batch_lines));
    accum.reserve(static_cast<size_t>(batch_lines));

    for (int64_t block = start; block < end; block += batch_lines) {
      const int B = static_cast<int>(std::min<int64_t>(batch_lines, end - block));
      const size_t Bs = static_cast<size_t>(B);
      coeff.resize(static_cast<size_t>(N) * Bs);

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpGather);
        for (int b = 0; b < B; ++b) {
          int64_t in_off = 0;
          int64_t out_off = 0;
          axis_pass_line_offsets(
              block + b,
              axis,
              bases,
              in_shape,
              in_strides,
              out_strides,
              idx,
              in_off,
              out_off);
          in_offsets[static_cast<size_t>(b)] = in_off;
          out_offsets[static_cast<size_t>(b)] = out_off;
        }
        if (scaled_gather_prefilter) {
          gather_axis_block_scaled(
              in,
              coeff.data(),
              Bs,
              B,
              N,
              in_offsets.data(),
              axis_stride_in,
              row_major_gather,
              strided_offset_gather,
              gather_scale);
        } else {
          gather_axis_block(
              in,
              coeff.data(),
              Bs,
              B,
              N,
              in_offsets.data(),
              axis_stride_in,
              row_major_gather,
              strided_offset_gather);
        }
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpPrefilter);
        if (scaled_gather_prefilter) {
          apply_interpolation_poles_colmajor(coeff, B, N, p.interp_degree);
        } else {
          get_interpolation_coefficients_colmajor(coeff, B, N, p.interp_degree);
        }
      }

      if (axis_contig_out) {
        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpAccumulate);
          y.resize(static_cast<size_t>(outN) * Bs);
          accumulate_row_runs_colmajor_preset_set(
              coeff.data(),
              Bs,
              B,
              plan,
              max_support,
              y.data());
        }

        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpScatter);
          for (int b = 0; b < B; ++b) {
            const int64_t out_off = out_offsets[static_cast<size_t>(b)];
            Scalar* dst = out + out_off;
            for (int l = 0; l < outN; ++l) {
              dst[static_cast<size_t>(l)] =
                  static_cast<Scalar>(y[static_cast<size_t>(l) * Bs +
                                        static_cast<size_t>(b)]);
            }
          }
        }
      } else {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpAccumulateScatter);
        const int64_t stride = out_strides[static_cast<size_t>(axis)];
        const double* weights = plan.weights.data();

        auto run_buffered_accumulate_scatter = [&]() {
          accum.resize(Bs);
          for (const RowRun1D& run : plan.row_runs) {
            if (run.interior) {
              for (int l = run.begin; l < run.end; ++l) {
                accumulate_row_colmajor_preset_set(
                    coeff.data(),
                    Bs,
                    B,
                    plan,
                    weights,
                    max_support,
                    l,
                    true,
                    accum.data());
                for (int b = 0; b < B; ++b) {
                  out[out_offsets[static_cast<size_t>(b)] +
                      static_cast<int64_t>(l) * stride] =
                      static_cast<Scalar>(accum[static_cast<size_t>(b)]);
                }
              }
            } else {
              for (int l = run.begin; l < run.end; ++l) {
                accumulate_row_colmajor_preset_set(
                    coeff.data(),
                    Bs,
                    B,
                    plan,
                    weights,
                    max_support,
                    l,
                    false,
                    accum.data());
                for (int b = 0; b < B; ++b) {
                  out[out_offsets[static_cast<size_t>(b)] +
                      static_cast<int64_t>(l) * stride] =
                      static_cast<Scalar>(accum[static_cast<size_t>(b)]);
                }
              }
            }
          }
        };

        if (direct_axis1_scatter || direct_2d_axis0_scatter) {
          if constexpr (std::is_same_v<Scalar, double>) {
            const bool unit_stride_offsets =
                offsets_are_unit_stride(out_offsets.data(), B);
            for (const RowRun1D& run : plan.row_runs) {
              const bool interior = run.interior != 0;
              for (int l = run.begin; l < run.end; ++l) {
                accumulate_scatter_row_colmajor_preset_set(
                    coeff.data(),
                    Bs,
                    B,
                    plan,
                    weights,
                    max_support,
                    l,
                    interior,
                    out,
                    out_offsets.data(),
                    stride,
                    unit_stride_offsets);
              }
            }
          } else {
            run_buffered_accumulate_scatter();
          }
        } else {
          run_buffered_accumulate_scatter();
        }
      }
    }
  };

  run_parallel_for_shape(nlines, plan, in_shape, worker);
}

template <typename Scalar>
static void resize_along_axis_batched_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  int axis,
  const LSParams& p,
  const Plan1D& plan,
  const std::vector<int>& bases,
  int64_t nlines)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionTotal);

  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int out_total = plan.out_total;
  const int corr_degree = (p.analy_degree < 0)
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);
  const int batch_lines =
      std::max(1, adaptive_batch_lines_for(
          in_shape, p, plan, nlines, false));
  const int64_t axis_stride_in = in_strides[static_cast<size_t>(axis)];
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const bool row_major_gather = row_gather_enabled();
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p, plan)
                        : 0;
  const bool fused_avg_restore =
      fused_projection_average_restore_enabled_for(nlines, plan);
  const bool scaled_output_prefilter =
      projection_output_prefilter_scale_enabled_for(nlines) &&
      p.analy_degree >= 0 &&
      corr_degree > 1 &&
      out_total > 1;
  const double output_prefilter_scale = scaled_output_prefilter
                                      ? interpolation_prefilter_lambda(corr_degree)
                                      : 1.0;
  const bool scaled_gather_prefilter =
      gather_prefilter_scale_enabled() && p.interp_degree > 1 && N > 1;
  const double gather_scale = scaled_gather_prefilter
                            ? interpolation_prefilter_lambda(p.interp_degree)
                            : 1.0;

  auto worker = [&](int64_t start, int64_t end) {
    std::vector<int64_t> idx(D, 0);
    std::vector<int64_t> in_offsets(static_cast<size_t>(batch_lines));
    std::vector<int64_t> out_offsets(static_cast<size_t>(batch_lines));
    std::vector<double> coeff;
    std::vector<double> y;
    std::vector<double> average;
    std::vector<double> filter_work;
    coeff.reserve(static_cast<size_t>(N) * static_cast<size_t>(batch_lines));
    y.reserve(static_cast<size_t>(out_total) * static_cast<size_t>(batch_lines));
    average.reserve(static_cast<size_t>(batch_lines));
    filter_work.reserve(static_cast<size_t>(std::max(out_total, N)) *
                        static_cast<size_t>(batch_lines));

    for (int64_t block = start; block < end; block += batch_lines) {
      const int B = static_cast<int>(std::min<int64_t>(batch_lines, end - block));
      const size_t Bs = static_cast<size_t>(B);
      coeff.resize(static_cast<size_t>(N) * Bs);
      y.resize(static_cast<size_t>(out_total) * Bs);

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionGather);
        for (int b = 0; b < B; ++b) {
          int64_t in_off = 0;
          int64_t out_off = 0;
          axis_pass_line_offsets(
              block + b,
              axis,
              bases,
              in_shape,
              in_strides,
              out_strides,
              idx,
              in_off,
              out_off);
          in_offsets[static_cast<size_t>(b)] = in_off;
          out_offsets[static_cast<size_t>(b)] = out_off;
        }
        if (scaled_gather_prefilter) {
          gather_axis_block_scaled(
              in,
              coeff.data(),
              Bs,
              B,
              N,
              in_offsets.data(),
              axis_stride_in,
              row_major_gather,
              false,
              gather_scale);
        } else {
          gather_axis_block(
              in,
              coeff.data(),
              Bs,
              B,
              N,
              in_offsets.data(),
              axis_stride_in,
              row_major_gather,
              false);
        }
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionInputPrefilter);
        if (scaled_gather_prefilter) {
          apply_interpolation_poles_colmajor(coeff, B, N, p.interp_degree);
        } else {
          get_interpolation_coefficients_colmajor(coeff, B, N, p.interp_degree);
        }
      }
      if (p.analy_degree >= 0 && !plan.direct_projection) {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionIntegrate);
        do_integ_colmajor(
            coeff,
            B,
            N,
            p.analy_degree + 1,
            average,
            filter_work);
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionAccumulate);
        accumulate_row_runs_colmajor_preset_set(
            coeff.data(),
            Bs,
            B,
            plan,
            max_support,
            y.data());
      }

      if (p.analy_degree >= 0) {
        if (!plan.direct_projection) {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionDiff);
          if (scaled_output_prefilter) {
            if (fused_avg_restore) {
              do_diff_colmajor_add_average_scaled(
                  y,
                  B,
                  out_total,
                  p.analy_degree + 1,
                  average,
                  output_prefilter_scale,
                  filter_work);
            } else {
              do_diff_colmajor(y, B, out_total, p.analy_degree + 1, filter_work);
              for (int l = 0; l < out_total; ++l) {
                double* y_col = y.data() + static_cast<size_t>(l) * Bs;
                for (int b = 0; b < B; ++b) {
                  const size_t bi = static_cast<size_t>(b);
                  y_col[bi] = (y_col[bi] + average[bi]) *
                              output_prefilter_scale;
                }
              }
            }
          } else if (fused_avg_restore) {
            do_diff_colmajor_add_average(
                y, B, out_total, p.analy_degree + 1, average, filter_work);
          } else {
            do_diff_colmajor(y, B, out_total, p.analy_degree + 1, filter_work);
            for (int l = 0; l < out_total; ++l) {
              double* y_col = y.data() + static_cast<size_t>(l) * Bs;
              for (int b = 0; b < B; ++b) {
                y_col[static_cast<size_t>(b)] += average[static_cast<size_t>(b)];
              }
            }
          }
        } else if (scaled_output_prefilter) {
          // The scaled output-prefilter route normally folds lambda into the
          // difference/average restoration. Direct cross-Gram rows have no
          // such stage, so apply the same normalization explicitly.
          for (double& value : y) {
            value *= output_prefilter_scale;
          }
        }
        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionOutputPrefilter);
          if (scaled_output_prefilter) {
            apply_interpolation_poles_colmajor(y, B, out_total, corr_degree);
          } else {
            get_interpolation_coefficients_colmajor(y, B, out_total, corr_degree);
          }
        }
        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionSampling);
          get_samples_colmajor(y, B, out_total, p.synthe_degree, filter_work);
        }
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionScatter);
        for (int b = 0; b < B; ++b) {
          const int64_t out_off = out_offsets[static_cast<size_t>(b)];
          if (axis_contig_out) {
            Scalar* dst = out + out_off;
            for (int l = 0; l < outN; ++l) {
              dst[static_cast<size_t>(l)] =
                  static_cast<Scalar>(y[static_cast<size_t>(l) * Bs +
                                        static_cast<size_t>(b)]);
            }
          } else {
            const int64_t stride = out_strides[static_cast<size_t>(axis)];
            for (int l = 0; l < outN; ++l) {
              out[out_off + static_cast<int64_t>(l) * stride] =
                  static_cast<Scalar>(y[static_cast<size_t>(l) * Bs +
                                        static_cast<size_t>(b)]);
            }
          }
        }
      }
    }
  };

  run_parallel_for_shape(nlines, plan, in_shape, worker);
}

static void resize_along_axis_batched_interp_f32_internal(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  int axis,
  const LSParams& p,
  const Plan1D& plan,
  const std::vector<int>& bases,
  int64_t nlines)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpTotal);

  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int batch_lines =
      std::max(1, adaptive_batch_lines_for(
          in_shape, p, plan, nlines, true));
  const int64_t axis_stride_in = in_strides[static_cast<size_t>(axis)];
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const bool row_major_gather = row_gather_enabled();
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p, plan)
                        : 0;
  const bool direct_axis1_scatter =
      should_use_direct_3d_axis1_scatter(in_shape, p, plan, axis, nlines, outN);
  const bool direct_2d_axis0_scatter =
      should_use_direct_2d_axis0_scatter(in_shape, p, axis, nlines, outN);
  const bool scaled_gather_prefilter =
      gather_prefilter_scale_enabled() && p.interp_degree > 1 && N > 1;
  const float gather_scale = scaled_gather_prefilter
                           ? interpolation_prefilter_lambda_f32(p.interp_degree)
                           : 1.0f;
  const bool strided_offset_gather =
      strided_offset_gather_enabled() &&
      (in_shape.size() == 2 ||
       *std::max_element(in_shape.begin(), in_shape.end()) >= 192);

  auto worker = [&](int64_t start, int64_t end) {
    std::vector<int64_t> idx(D, 0);
    std::vector<int64_t> in_offsets(static_cast<size_t>(batch_lines));
    std::vector<int64_t> out_offsets(static_cast<size_t>(batch_lines));
    std::vector<float> coeff;
    std::vector<float> y;
    std::vector<float> accum;
    coeff.reserve(static_cast<size_t>(N) * static_cast<size_t>(batch_lines));
    y.reserve(static_cast<size_t>(outN) * static_cast<size_t>(batch_lines));
    accum.reserve(static_cast<size_t>(batch_lines));

    for (int64_t block = start; block < end; block += batch_lines) {
      const int B = static_cast<int>(std::min<int64_t>(batch_lines, end - block));
      const size_t Bs = static_cast<size_t>(B);
      coeff.resize(static_cast<size_t>(N) * Bs);

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpGather);
        for (int b = 0; b < B; ++b) {
          int64_t in_off = 0;
          int64_t out_off = 0;
          axis_pass_line_offsets(
              block + b,
              axis,
              bases,
              in_shape,
              in_strides,
              out_strides,
              idx,
              in_off,
              out_off);
          in_offsets[static_cast<size_t>(b)] = in_off;
          out_offsets[static_cast<size_t>(b)] = out_off;
        }
        if (scaled_gather_prefilter) {
          gather_axis_block_scaled(
              in,
              coeff.data(),
              Bs,
              B,
              N,
              in_offsets.data(),
              axis_stride_in,
              row_major_gather,
              strided_offset_gather,
              gather_scale);
        } else {
          gather_axis_block(
              in,
              coeff.data(),
              Bs,
              B,
              N,
              in_offsets.data(),
              axis_stride_in,
              row_major_gather,
              strided_offset_gather);
        }
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpPrefilter);
        if (scaled_gather_prefilter) {
          apply_interpolation_poles_colmajor_f32(
              coeff, B, N, p.interp_degree);
        } else {
          get_interpolation_coefficients_colmajor_f32(
              coeff, B, N, p.interp_degree);
        }
      }

      if (axis_contig_out) {
        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpAccumulate);
          y.resize(static_cast<size_t>(outN) * Bs);
          accumulate_row_runs_colmajor_f32_preset_set(
              coeff.data(),
              Bs,
              B,
              plan,
              max_support,
              y.data());
        }

        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpScatter);
          for (int b = 0; b < B; ++b) {
            const int64_t out_off = out_offsets[static_cast<size_t>(b)];
            float* dst = out + out_off;
            for (int l = 0; l < outN; ++l) {
              dst[static_cast<size_t>(l)] =
                  y[static_cast<size_t>(l) * Bs + static_cast<size_t>(b)];
            }
          }
        }
      } else {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedInterpAccumulateScatter);
        const int64_t stride = out_strides[static_cast<size_t>(axis)];
        const double* weights = plan.weights.data();

        if (direct_axis1_scatter || direct_2d_axis0_scatter) {
          const bool unit_stride_offsets =
              offsets_are_unit_stride(out_offsets.data(), B);
          for (const RowRun1D& run : plan.row_runs) {
            const bool interior = run.interior != 0;
            for (int l = run.begin; l < run.end; ++l) {
              accumulate_scatter_row_colmajor_preset_set(
                  coeff.data(),
                  Bs,
                  B,
                  plan,
                  weights,
                  max_support,
                  l,
                  interior,
                  out,
                  out_offsets.data(),
                  stride,
                  unit_stride_offsets);
            }
          }
        } else {
          accum.resize(Bs);
          for (const RowRun1D& run : plan.row_runs) {
            for (int l = run.begin; l < run.end; ++l) {
              accumulate_row_colmajor_f32_preset_set(
                  coeff.data(),
                  Bs,
                  B,
                  plan,
                  weights,
                  max_support,
                  l,
                  run.interior != 0,
                  accum.data());
              for (int b = 0; b < B; ++b) {
                out[out_offsets[static_cast<size_t>(b)] +
                    static_cast<int64_t>(l) * stride] =
                    accum[static_cast<size_t>(b)];
              }
            }
          }
        }
      }
    }
  };

  run_parallel_for_shape(nlines, plan, in_shape, worker);
}

static void resize_along_axis_batched_f32_internal(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  int axis,
  const LSParams& p,
  const Plan1D& plan,
  const std::vector<int>& bases,
  int64_t nlines)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionTotal);

  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int out_total = plan.out_total;
  const int corr_degree = (p.analy_degree < 0)
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);
  const int batch_lines =
      std::max(1, adaptive_batch_lines_for(
          in_shape, p, plan, nlines, true));
  const int64_t axis_stride_in = in_strides[static_cast<size_t>(axis)];
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const bool row_major_gather = row_gather_enabled();
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p, plan)
                        : 0;
  const bool fused_avg_restore =
      fused_projection_average_restore_enabled_for(nlines, plan);
  const bool scaled_output_prefilter =
      projection_output_prefilter_scale_enabled_for(nlines) &&
      p.analy_degree >= 0 &&
      corr_degree > 1 &&
      out_total > 1;
  const float output_prefilter_scale = scaled_output_prefilter
                                     ? interpolation_prefilter_lambda_f32(corr_degree)
                                     : 1.0f;

  auto worker = [&](int64_t start, int64_t end) {
    std::vector<int64_t> idx(D, 0);
    std::vector<int64_t> in_offsets(static_cast<size_t>(batch_lines));
    std::vector<int64_t> out_offsets(static_cast<size_t>(batch_lines));
    std::vector<float> coeff;
    std::vector<float> y;
    std::vector<float> average;
    std::vector<float> filter_work;
    std::vector<float> line_dc;
    coeff.reserve(static_cast<size_t>(N) * static_cast<size_t>(batch_lines));
    y.reserve(static_cast<size_t>(out_total) * static_cast<size_t>(batch_lines));
    average.reserve(static_cast<size_t>(batch_lines));
    filter_work.reserve(static_cast<size_t>(std::max(out_total, N)) *
                        static_cast<size_t>(batch_lines));
    line_dc.reserve(static_cast<size_t>(batch_lines));

    for (int64_t block = start; block < end; block += batch_lines) {
      const int B = static_cast<int>(std::min<int64_t>(batch_lines, end - block));
      const size_t Bs = static_cast<size_t>(B);
      coeff.resize(static_cast<size_t>(N) * Bs);
      y.resize(static_cast<size_t>(out_total) * Bs);
      line_dc.resize(Bs);

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionGather);
        for (int b = 0; b < B; ++b) {
          int64_t in_off = 0;
          int64_t out_off = 0;
          axis_pass_line_offsets(
              block + b,
              axis,
              bases,
              in_shape,
              in_strides,
              out_strides,
              idx,
              in_off,
              out_off);
          in_offsets[static_cast<size_t>(b)] = in_off;
          out_offsets[static_cast<size_t>(b)] = out_off;
        }
        gather_axis_block(
            in,
            coeff.data(),
            Bs,
            B,
            N,
            in_offsets.data(),
            axis_stride_in,
            row_major_gather,
            false);
      }

      // The projection pipeline is linear and should preserve constants. Run
      // it on a DC-centered residual so float32 recursive filters do not turn a
      // constant line into small boundary drift.
      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionDcCenter);
        bool has_dc = false;
        for (int b = 0; b < B; ++b) {
          const size_t bi = static_cast<size_t>(b);
          float dc = coeff[bi];
          if (!std::isfinite(dc)) {
            dc = 0.0f;
          }
          line_dc[bi] = dc;
          has_dc = has_dc || (dc != 0.0f);
        }
        if (has_dc) {
          for (int n = 0; n < N; ++n) {
            float* col = coeff.data() + static_cast<size_t>(n) * Bs;
            for (int b = 0; b < B; ++b) {
              const size_t bi = static_cast<size_t>(b);
              col[bi] -= line_dc[bi];
            }
          }
        }
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionInputPrefilter);
        get_interpolation_coefficients_colmajor_f32(
            coeff, B, N, p.interp_degree);
      }
      if (p.analy_degree >= 0 && !plan.direct_projection) {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionIntegrate);
        do_integ_colmajor_f32(
            coeff,
            B,
            N,
            p.analy_degree + 1,
            average,
            filter_work);
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionAccumulate);
        accumulate_row_runs_colmajor_f32_preset_set(
            coeff.data(),
            Bs,
            B,
            plan,
            max_support,
            y.data());
      }

      if (p.analy_degree >= 0) {
        if (!plan.direct_projection) {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionDiff);
          if (scaled_output_prefilter) {
            if (fused_avg_restore) {
              do_diff_colmajor_add_average_scaled_f32(
                  y,
                  B,
                  out_total,
                  p.analy_degree + 1,
                  average,
                  output_prefilter_scale,
                  filter_work);
            } else {
              do_diff_colmajor_f32(
                  y, B, out_total, p.analy_degree + 1, filter_work);
              for (int l = 0; l < out_total; ++l) {
                float* y_col = y.data() + static_cast<size_t>(l) * Bs;
                for (int b = 0; b < B; ++b) {
                  const size_t bi = static_cast<size_t>(b);
                  y_col[bi] = (y_col[bi] + average[bi]) *
                              output_prefilter_scale;
                }
              }
            }
          } else if (fused_avg_restore) {
            do_diff_colmajor_add_average_f32(
                y, B, out_total, p.analy_degree + 1, average, filter_work);
          } else {
            do_diff_colmajor_f32(y, B, out_total, p.analy_degree + 1, filter_work);
            for (int l = 0; l < out_total; ++l) {
              float* y_col = y.data() + static_cast<size_t>(l) * Bs;
              for (int b = 0; b < B; ++b) {
                y_col[static_cast<size_t>(b)] += average[static_cast<size_t>(b)];
              }
            }
          }
        } else if (scaled_output_prefilter) {
          for (float& value : y) {
            value *= output_prefilter_scale;
          }
        }
        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionOutputPrefilter);
          if (scaled_output_prefilter) {
            apply_interpolation_poles_colmajor_f32(y, B, out_total, corr_degree);
          } else {
            get_interpolation_coefficients_colmajor_f32(y, B, out_total, corr_degree);
          }
        }
        {
          LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionSampling);
          get_samples_colmajor_f32(y, B, out_total, p.synthe_degree, filter_work);
        }
      }

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::BatchedProjectionScatter);
        for (int b = 0; b < B; ++b) {
          const int64_t out_off = out_offsets[static_cast<size_t>(b)];
          const float dc = line_dc[static_cast<size_t>(b)];
          if (axis_contig_out) {
            float* dst = out + out_off;
            for (int l = 0; l < outN; ++l) {
              dst[static_cast<size_t>(l)] =
                  y[static_cast<size_t>(l) * Bs + static_cast<size_t>(b)] + dc;
            }
          } else {
            const int64_t stride = out_strides[static_cast<size_t>(axis)];
            for (int l = 0; l < outN; ++l) {
              out[out_off + static_cast<int64_t>(l) * stride] =
                  y[static_cast<size_t>(l) * Bs + static_cast<size_t>(b)] + dc;
            }
          }
        }
      }
    }
  };

  run_parallel_for_shape(nlines, plan, in_shape, worker);
}

// -----------------------------------------------------------------------------
// Templated ND axis kernel over storage scalar (float or double).
// All internal computation (Plan1D, Work1D, filters) remains in double.
// -----------------------------------------------------------------------------
template <typename Scalar>
static void resize_along_axis_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p,
  const Plan1D* preplanned = nullptr)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::NdAxisTotal);

  const int D = static_cast<int>(in_shape.size());
  const auto in_strides  = strides_from_shape(in_shape);
  const auto out_strides = strides_from_shape(out_shape);
  const int N_line = static_cast<int>(in_shape[static_cast<size_t>(axis)]);
  const int outN_line = static_cast<int>(out_shape[static_cast<size_t>(axis)]);

  // Early identity short-circuit on this axis:
  {
    const double eps = 1e-12;
    const bool identity_axis =
        (outN_line == N_line) &&
        (std::abs(p.shift) <= eps) &&
        (p.synthe_degree == p.interp_degree);
    if (identity_axis) {
      const int64_t total = prod_elems(in_shape); // in_shape == out_shape in this pass
      std::copy(in, in + total, out);             // Scalar -> Scalar
      return;
    }
  }

  // Total number of independent 1-D lines (all dims except 'axis')
  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) {
    if (d != axis) {
      nlines *= in_shape[static_cast<size_t>(d)];
    }
  }

  // List non-axis dimensions (rightmost fastest)
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) {
      bases.push_back(d);
    }
  }

  // Endpoint alignment has no scale when either endpoint grid is a
  // singleton. Keep these cases explicit, symmetric and DC-preserving:
  // replicate a singleton input; reduce a projected singleton output to the
  // arithmetic line mean. Degree-zero interpolation explicitly averages the
  // central pair at an even-length half-grid tie. Higher interpolation degrees
  // continue below, where Plan1D evaluates the spline at x=(N-1)/2.
  const bool projected_singleton =
      (outN_line == 1 && p.analy_degree >= 0);
  const bool nearest_center_singleton =
      (outN_line == 1 && p.analy_degree < 0 &&
       p.interp_degree == 0 && std::abs(p.shift) <= 1e-12);
  if (N_line == 1 || projected_singleton || nearest_center_singleton) {
    std::vector<int64_t> idx(static_cast<size_t>(D), 0);
    const bool replicate = (N_line == 1);

    for (int64_t line = 0; line < nlines; ++line) {
      std::fill(idx.begin(), idx.end(), 0);
      int64_t t = line;
      for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
        const int d = bases[static_cast<size_t>(bi)];
        idx[static_cast<size_t>(d)] =
            t % in_shape[static_cast<size_t>(d)];
        t /= in_shape[static_cast<size_t>(d)];
      }

      int64_t in_off = 0;
      int64_t out_off = 0;
      for (int d = 0; d < D; ++d) {
        if (d != axis) {
          in_off += idx[static_cast<size_t>(d)] *
                    in_strides[static_cast<size_t>(d)];
          out_off += idx[static_cast<size_t>(d)] *
                     out_strides[static_cast<size_t>(d)];
        }
      }

      if (replicate) {
        const Scalar value = in[in_off];
        const int64_t out_stride = out_strides[static_cast<size_t>(axis)];
        for (int l = 0; l < outN_line; ++l) {
          out[out_off + static_cast<int64_t>(l) * out_stride] = value;
        }
      } else if (projected_singleton) {
        const int64_t in_stride = in_strides[static_cast<size_t>(axis)];
        long double sum = 0.0L;
        for (int i = 0; i < N_line; ++i) {
          sum += static_cast<long double>(
              in[in_off + static_cast<int64_t>(i) * in_stride]);
        }
        out[out_off] = static_cast<Scalar>(
            sum / static_cast<long double>(N_line));
      } else {
        const int64_t in_stride = in_strides[static_cast<size_t>(axis)];
        const int left = (N_line - 1) / 2;
        const int right = N_line / 2;
        const long double center =
            (static_cast<long double>(
                 in[in_off + static_cast<int64_t>(left) * in_stride]) +
             static_cast<long double>(
                 in[in_off + static_cast<int64_t>(right) * in_stride])) *
            0.5L;
        out[out_off] = static_cast<Scalar>(center);
      }
    }
    return;
  }

  // A reusable N-D plan supplies its immutable Plan1D directly.  The local
  // shared_ptr keeps the ordinary cached path alive for the duration of this
  // pass without imposing a cache lookup on preplanned execution.
  std::shared_ptr<const Plan1D> plan_handle;
  if (preplanned == nullptr) {
    plan_handle = get_plan_1d_cached(N_line, p);
    preplanned = plan_handle.get();
  }
  const Plan1D& plan = *preplanned;

  const BatchedAxisMode axis_mode = batched_axis_mode();

  if (axis_mode != BatchedAxisMode::Off &&
      can_use_linear_interp_fast_path(in_shape, out_shape, axis, p)) {
    if constexpr (std::is_same_v<Scalar, float>) {
      if (!float32_internal_enabled()) {
        if (in_shape.size() == 2) {
          resize_along_axis_2d_linear_fast(
              in,
              out,
              in_shape,
              out_shape,
              axis,
              plan,
              nlines);
        } else {
          resize_along_axis_linear_direct(
              in,
              out,
              in_shape,
              in_strides,
              out_strides,
              axis,
              plan,
              bases,
              nlines);
        }
        return;
      }
    } else if constexpr (std::is_same_v<Scalar, double>) {
      if (in_shape.size() == 2) {
        resize_along_axis_2d_linear_fast(
            in,
            out,
            in_shape,
            out_shape,
            axis,
            plan,
            nlines);
      } else {
        resize_along_axis_linear_direct(
            in,
            out,
            in_shape,
            in_strides,
            out_strides,
            axis,
            plan,
            bases,
            nlines);
      }
      return;
    }
  }

  if (should_use_batched_axis(
          axis_mode,
          in_shape,
          p,
          plan,
          nlines)) {
    if constexpr (std::is_same_v<Scalar, float>) {
      const bool use_float32_internal =
          float32_internal_enabled_for(in_shape, p);
      if (use_float32_internal) {
        if (p.analy_degree < 0) {
          resize_along_axis_batched_interp_f32_internal(
              in,
              out,
              in_shape,
              in_strides,
              out_strides,
              axis,
              p,
              plan,
              bases,
              nlines);
          return;
        }
        resize_along_axis_batched_f32_internal(
            in,
            out,
            in_shape,
            in_strides,
            out_strides,
            axis,
            p,
            plan,
            bases,
            nlines);
        return;
      }
    }
    if (p.analy_degree < 0) {
      resize_along_axis_batched_interp_t(
          in,
          out,
          in_shape,
          in_strides,
          out_strides,
          axis,
          p,
          plan,
          bases,
          nlines);
      return;
    }
    resize_along_axis_batched_t(
        in,
        out,
        in_shape,
        in_strides,
        out_strides,
        axis,
        p,
        plan,
        bases,
        nlines);
    return;
  }

  // Worker that processes a range of 1-D lines [start, end)
  auto worker = [&](int64_t start, int64_t end) {
    Work1D workspace; // per-thread reusable workspace (double internal)
    std::vector<int64_t> idx(D, 0);
    std::vector<double>  line_out;
    workspace.line.reserve(static_cast<size_t>(N_line));
    workspace.ext_full.reserve(static_cast<size_t>(plan.left_pad +
                                            plan.length_total +
                                            plan.right_pad));
    workspace.y.reserve(static_cast<size_t>(plan.out_total));
    line_out.reserve(static_cast<size_t>(plan.outN));

    const bool axis_contig_in  =
        (in_strides[static_cast<size_t>(axis)] == 1);
    const bool axis_contig_out =
        (out_strides[static_cast<size_t>(axis)] == 1);

    for (int64_t line = start; line < end; ++line) {
      int64_t in_off  = 0;
      int64_t out_off = 0;
      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::LineFallbackOffset);
        std::fill(idx.begin(), idx.end(), 0);

        // Unravel 'line' into coordinates for all dims except 'axis'
        int64_t t = line;
        for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
          const int d = bases[static_cast<size_t>(bi)];
          idx[static_cast<size_t>(d)] =
              t % in_shape[static_cast<size_t>(d)];
          t /= in_shape[static_cast<size_t>(d)];
        }

        // Offsets at the start of this line
        for (int d = 0; d < D; ++d) {
          if (d != axis) {
            in_off  += idx[static_cast<size_t>(d)] *
                       in_strides[static_cast<size_t>(d)];
            out_off += idx[static_cast<size_t>(d)] *
                       out_strides[static_cast<size_t>(d)];
          }
        }
      }

      // --- Fast path: axis contiguous in both in & out and Scalar == double ---
      if constexpr (std::is_same_v<Scalar, double>) {
        if (axis_contig_in && axis_contig_out) {
          // Direct 1-D resize on raw buffers, no gather/scatter via vectors.
          {
            LSRESIZE_PROFILE_SCOPE(profile::Phase::LineFallbackResize1D);
            resize_1d_line_contiguous(in + in_off, out + out_off, p, plan, workspace);
          }
          continue;
        }
      }

      // --- Fallback: gather into workspace.line (double), run 1-D core from line ---

      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::LineFallbackGather);
        workspace.line.resize(static_cast<size_t>(N_line));

        if (axis_contig_in) {
          // contiguous axis: simple block copy with cast
          const Scalar* in_line = in + in_off;
          for (int i = 0; i < N_line; ++i) {
            workspace.line[static_cast<size_t>(i)] =
              static_cast<double>(in_line[static_cast<size_t>(i)]);
          }
        } else {
          // non-contiguous axis: strided gather with cast
          const int64_t stride = in_strides[static_cast<size_t>(axis)];
          for (int i = 0; i < N_line; ++i) {
            workspace.line[static_cast<size_t>(i)] =
                static_cast<double>(
                    in[in_off +
                       static_cast<int64_t>(i) * stride]);
          }
        }
      }

      // Fast planned path with workspace reuse (double internal)
      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::LineFallbackResize1D);
        resize_1d_line_buffered(workspace.line, line_out, p, plan, workspace);
      }

      // Scatter to output (Scalar storage)
      {
        LSRESIZE_PROFILE_SCOPE(profile::Phase::LineFallbackScatter);
        const bool contig_out = axis_contig_out;
        if (contig_out) {
          if constexpr (std::is_same_v<Scalar, double>) {
            // One-shot block write when the axis is contiguous
            std::memcpy(out + out_off,
                        line_out.data(),
                        line_out.size() * sizeof(double));
          } else {
            for (size_t i = 0; i < line_out.size(); ++i) {
              out[out_off + static_cast<int64_t>(i)] =
                  static_cast<Scalar>(line_out[static_cast<size_t>(i)]);
            }
          }
        } else {
          for (int64_t i = 0;
               i < static_cast<int64_t>(line_out.size());
               ++i) {
            out[out_off +
                i * out_strides[static_cast<size_t>(axis)]] =
                static_cast<Scalar>(
                    line_out[static_cast<size_t>(i)]);
          }
        }
      }
    }
  };

  // Centralized scheduling: OpenMP, std::thread, or serial
  {
    LSRESIZE_PROFILE_SCOPE(profile::Phase::LineFallbackTotal);
    run_parallel_for_shape(nlines, plan, in_shape, worker);
  }
}

#if LSRESIZE_GNU_X86_TARGETS
__attribute__((target("avx2,fma")))
static void resize_2d_linear_row_n2_col2_f32_avx2(
  const float* LS_RESTRICT row0,
  const float* LS_RESTRICT row1,
  float* LS_RESTRICT dst,
  int64_t begin,
  int64_t end,
  const int* LS_RESTRICT cs0,
  const int* LS_RESTRICT cs1,
  const float* LS_RESTRICT cw0,
  const float* LS_RESTRICT cw1,
  float wr0,
  float wr1)
{
  const __m256 vwr0 = _mm256_set1_ps(wr0);
  const __m256 vwr1 = _mm256_set1_ps(wr1);
  int64_t c = begin;
  for (; c + 8 <= end; c += 8) {
    const size_t ci = static_cast<size_t>(c);
    const __m256i idx0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cs0 + ci));
    const __m256i idx1 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cs1 + ci));
    const __m256 wc0v = _mm256_loadu_ps(cw0 + ci);
    const __m256 wc1v = _mm256_loadu_ps(cw1 + ci);

    const __m256 r00 = _mm256_i32gather_ps(row0, idx0, 4);
    const __m256 r01 = _mm256_i32gather_ps(row0, idx1, 4);
    const __m256 r10 = _mm256_i32gather_ps(row1, idx0, 4);
    const __m256 r11 = _mm256_i32gather_ps(row1, idx1, 4);

    const __m256 h0 = _mm256_fmadd_ps(wc1v, r01, _mm256_mul_ps(wc0v, r00));
    const __m256 h1 = _mm256_fmadd_ps(wc1v, r11, _mm256_mul_ps(wc0v, r10));
    const __m256 acc = _mm256_fmadd_ps(vwr1, h1, _mm256_mul_ps(vwr0, h0));
    _mm256_storeu_ps(dst + ci, acc);
  }

  for (; c < end; ++c) {
    const size_t ci = static_cast<size_t>(c);
    const int c0 = cs0[ci];
    const int c1 = cs1[ci];
    const float h0 =
        cw0[ci] * row0[static_cast<size_t>(c0)] +
        cw1[ci] * row0[static_cast<size_t>(c1)];
    const float h1 =
        cw0[ci] * row1[static_cast<size_t>(c0)] +
        cw1[ci] * row1[static_cast<size_t>(c1)];
    dst[ci] = wr0 * h0 + wr1 * h1;
  }
}

__attribute__((target("avx2,fma")))
static void resize_2d_linear_row_n2_col2_f64_avx2(
  const double* LS_RESTRICT row0,
  const double* LS_RESTRICT row1,
  double* LS_RESTRICT dst,
  int64_t begin,
  int64_t end,
  const int* LS_RESTRICT cs0,
  const int* LS_RESTRICT cs1,
  const double* LS_RESTRICT cw0,
  const double* LS_RESTRICT cw1,
  double wr0,
  double wr1)
{
  const __m256d vwr0 = _mm256_set1_pd(wr0);
  const __m256d vwr1 = _mm256_set1_pd(wr1);
  int64_t c = begin;
  for (; c + 4 <= end; c += 4) {
    const size_t ci = static_cast<size_t>(c);
    const __m128i idx0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(cs0 + ci));
    const __m128i idx1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(cs1 + ci));
    const __m256d wc0v = _mm256_loadu_pd(cw0 + ci);
    const __m256d wc1v = _mm256_loadu_pd(cw1 + ci);

    const __m256d r00 = _mm256_i32gather_pd(row0, idx0, 8);
    const __m256d r01 = _mm256_i32gather_pd(row0, idx1, 8);
    const __m256d r10 = _mm256_i32gather_pd(row1, idx0, 8);
    const __m256d r11 = _mm256_i32gather_pd(row1, idx1, 8);

    const __m256d h0 = _mm256_fmadd_pd(wc1v, r01, _mm256_mul_pd(wc0v, r00));
    const __m256d h1 = _mm256_fmadd_pd(wc1v, r11, _mm256_mul_pd(wc0v, r10));
    const __m256d acc = _mm256_fmadd_pd(vwr1, h1, _mm256_mul_pd(vwr0, h0));
    _mm256_storeu_pd(dst + ci, acc);
  }

  for (; c < end; ++c) {
    const size_t ci = static_cast<size_t>(c);
    const int c0 = cs0[ci];
    const int c1 = cs1[ci];
    const double h0 =
        cw0[ci] * row0[static_cast<size_t>(c0)] +
        cw1[ci] * row0[static_cast<size_t>(c1)];
    const double h1 =
        cw0[ci] * row1[static_cast<size_t>(c0)] +
        cw1[ci] * row1[static_cast<size_t>(c1)];
    dst[ci] = wr0 * h0 + wr1 * h1;
  }
}
#endif

template <typename Scalar>
static void resize_2d_linear_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const Plan1D* preplanned0 = nullptr,
  const Plan1D* preplanned1 = nullptr)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::Fused2DLinearTotal);

  const int64_t in_h = in_shape[0];
  const int64_t in_w = in_shape[1];
  const int64_t out_h = out_shape[0];
  const int64_t out_w = out_shape[1];
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;

  std::shared_ptr<const Plan1D> row_plan_handle;
  std::shared_ptr<const Plan1D> col_plan_handle;
  if (preplanned0 == nullptr) {
    row_plan_handle = get_plan_1d_cached(static_cast<int>(in_h), p0);
    preplanned0 = row_plan_handle.get();
  }
  if (preplanned1 == nullptr) {
    col_plan_handle = get_plan_1d_cached(static_cast<int>(in_w), p1);
    preplanned1 = col_plan_handle.get();
  }
  const Plan1D& row_plan = *preplanned0;
  const Plan1D& col_plan = *preplanned1;

  const auto row_direct = direct_linear_plan_view<Accum>(row_plan);
  const auto col_direct = direct_linear_plan_view<Accum>(col_plan);
  if (!row_direct.ok || !col_direct.ok) {
    std::vector<Scalar> tmp(
        static_cast<size_t>(out_h) * static_cast<size_t>(in_w));
    std::vector<int64_t> mid_shape = {out_h, in_w};
    resize_along_axis_2d_linear_direct(
        in, tmp.data(), in_shape, mid_shape, 0, row_plan, in_w);
    resize_along_axis_2d_linear_direct(
        tmp.data(), out, mid_shape, out_shape, 1, col_plan, out_h);
    return;
  }

  const unsigned char* LS_RESTRICT rc = row_direct.count;
  const int* LS_RESTRICT rs0 = row_direct.src0;
  const int* LS_RESTRICT rs1 = row_direct.src1;
  const int* LS_RESTRICT rs2 = row_direct.src2;
  const Accum* LS_RESTRICT rw0 = row_direct.w0;
  const Accum* LS_RESTRICT rw1 = row_direct.w1;
  const Accum* LS_RESTRICT rw2 = row_direct.w2;
  const unsigned char* LS_RESTRICT cc = col_direct.count;
  const int* LS_RESTRICT cs0 = col_direct.src0;
  const int* LS_RESTRICT cs1 = col_direct.src1;
  const int* LS_RESTRICT cs2 = col_direct.src2;
  const Accum* LS_RESTRICT cw0 = col_direct.w0;
  const Accum* LS_RESTRICT cw1 = col_direct.w1;
  const Accum* LS_RESTRICT cw2 = col_direct.w2;
  const auto col2_run = longest_count_run(cc, out_w, 2);
  const int64_t col2_begin = col2_run.first;
  const int64_t col2_end = col2_run.second;
  const bool all_axes_grow = (out_h > in_h) && (out_w > in_w);
  const bool use_avx2_linear =
      all_axes_grow &&
      avx2_linear_enabled() &&
      (col2_end - col2_begin >= 16);

  auto worker = [&](int64_t start, int64_t end) {
    for (int64_t r = start; r < end; ++r) {
      const size_t ri = static_cast<size_t>(r);
      const Scalar* LS_RESTRICT row0 =
          in + static_cast<int64_t>(rs0[ri]) * in_w;
      const Scalar* LS_RESTRICT row1 =
          in + static_cast<int64_t>(rs1[ri]) * in_w;
      const Scalar* LS_RESTRICT row2 =
          in + static_cast<int64_t>(rs2[ri]) * in_w;
      const int nr = static_cast<int>(rc[ri]);
      Scalar* LS_RESTRICT dst = out + r * out_w;
      const Accum wr0 = rw0[ri];
      const Accum wr1 = rw1[ri];
      const Accum wr2 = rw2[ri];

      if (nr == 2) {
        auto scalar_n2 = [&](int64_t c_begin, int64_t c_end) {
          for (int64_t c = c_begin; c < c_end; ++c) {
            const size_t ci = static_cast<size_t>(c);
            const int c0 = cs0[ci];
            const int c1 = cs1[ci];
            const int c2 = cs2[ci];
            const Accum wc0 = cw0[ci];
            const Accum wc1 = cw1[ci];
            const Accum wc2 = cw2[ci];
            Accum acc = Accum(0);
            switch (cc[ci]) {
              case 3:
                acc = wr0 * (
                    wc0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                    wc1 * static_cast<Accum>(row0[static_cast<size_t>(c1)]) +
                    wc2 * static_cast<Accum>(row0[static_cast<size_t>(c2)]));
                acc += wr1 * (
                    wc0 * static_cast<Accum>(row1[static_cast<size_t>(c0)]) +
                    wc1 * static_cast<Accum>(row1[static_cast<size_t>(c1)]) +
                    wc2 * static_cast<Accum>(row1[static_cast<size_t>(c2)]));
                break;
              case 2:
                acc = wr0 * (
                    wc0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                    wc1 * static_cast<Accum>(row0[static_cast<size_t>(c1)]));
                acc += wr1 * (
                    wc0 * static_cast<Accum>(row1[static_cast<size_t>(c0)]) +
                    wc1 * static_cast<Accum>(row1[static_cast<size_t>(c1)]));
                break;
              case 1:
                acc = wc0 * (
                    wr0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                    wr1 * static_cast<Accum>(row1[static_cast<size_t>(c0)]));
                break;
              default:
                break;
            }
            dst[ci] = static_cast<Scalar>(acc);
          }
        };

#if LSRESIZE_GNU_X86_TARGETS
        if (use_avx2_linear) {
          scalar_n2(0, col2_begin);
          if constexpr (std::is_same_v<Scalar, float>) {
            resize_2d_linear_row_n2_col2_f32_avx2(
                row0,
                row1,
                dst,
                col2_begin,
                col2_end,
                cs0,
                cs1,
                cw0,
                cw1,
                wr0,
                wr1);
          } else if constexpr (std::is_same_v<Scalar, double>) {
            resize_2d_linear_row_n2_col2_f64_avx2(
                row0,
                row1,
                dst,
                col2_begin,
                col2_end,
                cs0,
                cs1,
                cw0,
                cw1,
                wr0,
                wr1);
          }
          scalar_n2(col2_end, out_w);
          continue;
        }
#endif

        scalar_n2(0, out_w);
        continue;
      }

      if (nr == 1) {
        for (int64_t c = 0; c < out_w; ++c) {
          const size_t ci = static_cast<size_t>(c);
          const int c0 = cs0[ci];
          const int c1 = cs1[ci];
          const int c2 = cs2[ci];
          const Accum wc0 = cw0[ci];
          const Accum wc1 = cw1[ci];
          const Accum wc2 = cw2[ci];
          Accum acc = Accum(0);
          switch (cc[ci]) {
            case 3:
              acc = wr0 * (
                  wc0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row0[static_cast<size_t>(c1)]) +
                  wc2 * static_cast<Accum>(row0[static_cast<size_t>(c2)]));
              break;
            case 2:
              acc = wr0 * (
                  wc0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row0[static_cast<size_t>(c1)]));
              break;
            case 1:
              acc = wr0 * wc0 *
                    static_cast<Accum>(row0[static_cast<size_t>(c0)]);
              break;
            default:
              break;
          }
          dst[ci] = static_cast<Scalar>(acc);
        }
        continue;
      }

      if (nr == 3) {
        for (int64_t c = 0; c < out_w; ++c) {
          const size_t ci = static_cast<size_t>(c);
          const int c0 = cs0[ci];
          const int c1 = cs1[ci];
          const int c2 = cs2[ci];
          const Accum wc0 = cw0[ci];
          const Accum wc1 = cw1[ci];
          const Accum wc2 = cw2[ci];
          Accum acc = Accum(0);
          switch (cc[ci]) {
            case 3:
              acc = wr0 * (
                  wc0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row0[static_cast<size_t>(c1)]) +
                  wc2 * static_cast<Accum>(row0[static_cast<size_t>(c2)]));
              acc += wr1 * (
                  wc0 * static_cast<Accum>(row1[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row1[static_cast<size_t>(c1)]) +
                  wc2 * static_cast<Accum>(row1[static_cast<size_t>(c2)]));
              acc += wr2 * (
                  wc0 * static_cast<Accum>(row2[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row2[static_cast<size_t>(c1)]) +
                  wc2 * static_cast<Accum>(row2[static_cast<size_t>(c2)]));
              break;
            case 2:
              acc = wr0 * (
                  wc0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row0[static_cast<size_t>(c1)]));
              acc += wr1 * (
                  wc0 * static_cast<Accum>(row1[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row1[static_cast<size_t>(c1)]));
              acc += wr2 * (
                  wc0 * static_cast<Accum>(row2[static_cast<size_t>(c0)]) +
                  wc1 * static_cast<Accum>(row2[static_cast<size_t>(c1)]));
              break;
            case 1:
              acc = wc0 * (
                  wr0 * static_cast<Accum>(row0[static_cast<size_t>(c0)]) +
                  wr1 * static_cast<Accum>(row1[static_cast<size_t>(c0)]) +
                  wr2 * static_cast<Accum>(row2[static_cast<size_t>(c0)]));
              break;
            default:
              break;
          }
          dst[ci] = static_cast<Scalar>(acc);
        }
        continue;
      }
    }
  };

  run_parallel_for_shape(out_h, row_plan, in_shape, worker);
}

template <typename Scalar>
static void resize_3d_linear_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const LSParams& p2,
  const Plan1D* preplanned0 = nullptr,
  const Plan1D* preplanned1 = nullptr,
  const Plan1D* preplanned2 = nullptr)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::Fused3DLinearTotal);

  const int64_t in_n0 = in_shape[0];
  const int64_t in_n1 = in_shape[1];
  const int64_t in_n2 = in_shape[2];
  const int64_t out_n0 = out_shape[0];
  const int64_t out_n1 = out_shape[1];
  const int64_t out_n2 = out_shape[2];
  const int64_t in_s0 = in_n1 * in_n2;
  const int64_t in_s1 = in_n2;
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;

  std::shared_ptr<const Plan1D> plan0_handle;
  std::shared_ptr<const Plan1D> plan1_handle;
  std::shared_ptr<const Plan1D> plan2_handle;
  if (preplanned0 == nullptr) {
    plan0_handle = get_plan_1d_cached(static_cast<int>(in_n0), p0);
    preplanned0 = plan0_handle.get();
  }
  if (preplanned1 == nullptr) {
    plan1_handle = get_plan_1d_cached(static_cast<int>(in_n1), p1);
    preplanned1 = plan1_handle.get();
  }
  if (preplanned2 == nullptr) {
    plan2_handle = get_plan_1d_cached(static_cast<int>(in_n2), p2);
    preplanned2 = plan2_handle.get();
  }
  const Plan1D& plan0 = *preplanned0;
  const Plan1D& plan1 = *preplanned1;
  const Plan1D& plan2 = *preplanned2;

  const auto direct0 = direct_linear_plan_view<Accum>(plan0);
  const auto direct1 = direct_linear_plan_view<Accum>(plan1);
  const auto direct2 = direct_linear_plan_view<Accum>(plan2);

  if (!direct0.ok || !direct1.ok || !direct2.ok) {
    std::vector<int64_t> shape1 = {out_n0, in_n1, in_n2};
    std::vector<int64_t> shape2 = {out_n0, out_n1, in_n2};
    std::vector<Scalar> tmp1(
        static_cast<size_t>(out_n0) *
        static_cast<size_t>(in_n1) *
        static_cast<size_t>(in_n2));
    std::vector<Scalar> tmp2(
        static_cast<size_t>(out_n0) *
        static_cast<size_t>(out_n1) *
        static_cast<size_t>(in_n2));
    resize_along_axis_t(in, tmp1.data(), in_shape, shape1, 0, p0, &plan0);
    resize_along_axis_t(
        tmp1.data(), tmp2.data(), shape1, shape2, 1, p1, &plan1);
    resize_along_axis_t(
        tmp2.data(), out, shape2, out_shape, 2, p2, &plan2);
    return;
  }

  const unsigned char* LS_RESTRICT c0 = direct0.count;
  const int* LS_RESTRICT s00 = direct0.src0;
  const int* LS_RESTRICT s01 = direct0.src1;
  const int* LS_RESTRICT s02 = direct0.src2;
  const Accum* LS_RESTRICT a00 = direct0.w0;
  const Accum* LS_RESTRICT a01 = direct0.w1;
  const Accum* LS_RESTRICT a02 = direct0.w2;
  const unsigned char* LS_RESTRICT c1 = direct1.count;
  const int* LS_RESTRICT s10 = direct1.src0;
  const int* LS_RESTRICT s11 = direct1.src1;
  const int* LS_RESTRICT s12 = direct1.src2;
  const Accum* LS_RESTRICT a10 = direct1.w0;
  const Accum* LS_RESTRICT a11 = direct1.w1;
  const Accum* LS_RESTRICT a12 = direct1.w2;
  const unsigned char* LS_RESTRICT c2 = direct2.count;
  const int* LS_RESTRICT s20 = direct2.src0;
  const int* LS_RESTRICT s21 = direct2.src1;
  const int* LS_RESTRICT s22 = direct2.src2;
  const Accum* LS_RESTRICT a20 = direct2.w0;
  const Accum* LS_RESTRICT a21 = direct2.w1;
  const Accum* LS_RESTRICT a22 = direct2.w2;

  auto worker = [&](int64_t start, int64_t end) {
    int src0[3];
    int src1[3];
    int src2[3];
    Accum wt0[3];
    Accum wt1[3];
    Accum wt2[3];

    for (int64_t line = start; line < end; ++line) {
      const int64_t o0 = line / out_n1;
      const int64_t o1 = line - o0 * out_n1;
      const size_t i0 = static_cast<size_t>(o0);
      const size_t i1 = static_cast<size_t>(o1);
      const int n0 = static_cast<int>(c0[i0]);
      const int n1 = static_cast<int>(c1[i1]);

      src0[0] = s00[i0];
      src0[1] = s01[i0];
      src0[2] = s02[i0];
      wt0[0] = a00[i0];
      wt0[1] = a01[i0];
      wt0[2] = a02[i0];
      src1[0] = s10[i1];
      src1[1] = s11[i1];
      src1[2] = s12[i1];
      wt1[0] = a10[i1];
      wt1[1] = a11[i1];
      wt1[2] = a12[i1];

      Scalar* LS_RESTRICT dst = out + line * out_n2;
      if (n0 == 2 && n1 == 2) {
        const Scalar* LS_RESTRICT p00 =
            in + static_cast<int64_t>(src0[0]) * in_s0 +
            static_cast<int64_t>(src1[0]) * in_s1;
        const Scalar* LS_RESTRICT p01 =
            in + static_cast<int64_t>(src0[0]) * in_s0 +
            static_cast<int64_t>(src1[1]) * in_s1;
        const Scalar* LS_RESTRICT p10 =
            in + static_cast<int64_t>(src0[1]) * in_s0 +
            static_cast<int64_t>(src1[0]) * in_s1;
        const Scalar* LS_RESTRICT p11 =
            in + static_cast<int64_t>(src0[1]) * in_s0 +
            static_cast<int64_t>(src1[1]) * in_s1;
        const Accum w00 = wt0[0] * wt1[0];
        const Accum w01 = wt0[0] * wt1[1];
        const Accum w10 = wt0[1] * wt1[0];
        const Accum w11 = wt0[1] * wt1[1];

        for (int64_t o2 = 0; o2 < out_n2; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          Accum acc = Accum(0);
          switch (c2[i2]) {
            case 3: {
              const int z0 = s20[i2];
              const int z1 = s21[i2];
              const int z2 = s22[i2];
              const Accum wz0 = a20[i2];
              const Accum wz1 = a21[i2];
              const Accum wz2 = a22[i2];
              acc = wz0 * (
                  w00 * static_cast<Accum>(p00[z0]) +
                  w01 * static_cast<Accum>(p01[z0]) +
                  w10 * static_cast<Accum>(p10[z0]) +
                  w11 * static_cast<Accum>(p11[z0]));
              acc += wz1 * (
                  w00 * static_cast<Accum>(p00[z1]) +
                  w01 * static_cast<Accum>(p01[z1]) +
                  w10 * static_cast<Accum>(p10[z1]) +
                  w11 * static_cast<Accum>(p11[z1]));
              acc += wz2 * (
                  w00 * static_cast<Accum>(p00[z2]) +
                  w01 * static_cast<Accum>(p01[z2]) +
                  w10 * static_cast<Accum>(p10[z2]) +
                  w11 * static_cast<Accum>(p11[z2]));
              break;
            }
            case 2: {
              const int z0 = s20[i2];
              const int z1 = s21[i2];
              const Accum wz0 = a20[i2];
              const Accum wz1 = a21[i2];
              acc = wz0 * (
                  w00 * static_cast<Accum>(p00[z0]) +
                  w01 * static_cast<Accum>(p01[z0]) +
                  w10 * static_cast<Accum>(p10[z0]) +
                  w11 * static_cast<Accum>(p11[z0]));
              acc += wz1 * (
                  w00 * static_cast<Accum>(p00[z1]) +
                  w01 * static_cast<Accum>(p01[z1]) +
                  w10 * static_cast<Accum>(p10[z1]) +
                  w11 * static_cast<Accum>(p11[z1]));
              break;
            }
            case 1: {
              const int z0 = s20[i2];
              const Accum wz0 = a20[i2];
              acc = wz0 * (
                  w00 * static_cast<Accum>(p00[z0]) +
                  w01 * static_cast<Accum>(p01[z0]) +
                  w10 * static_cast<Accum>(p10[z0]) +
                  w11 * static_cast<Accum>(p11[z0]));
              break;
            }
            default:
              break;
          }
          dst[i2] = static_cast<Scalar>(acc);
        }
        continue;
      }

      if (n0 == 1 && n1 == 2) {
        const Scalar* LS_RESTRICT p0 =
            in + static_cast<int64_t>(src0[0]) * in_s0 +
            static_cast<int64_t>(src1[0]) * in_s1;
        const Scalar* LS_RESTRICT p1 =
            in + static_cast<int64_t>(src0[0]) * in_s0 +
            static_cast<int64_t>(src1[1]) * in_s1;
        const Accum w0 = wt0[0] * wt1[0];
        const Accum w1 = wt0[0] * wt1[1];

        for (int64_t o2 = 0; o2 < out_n2; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          Accum acc = Accum(0);
          switch (c2[i2]) {
            case 3: {
              const int z0 = s20[i2];
              const int z1 = s21[i2];
              const int z2 = s22[i2];
              acc = a20[i2] * (
                  w0 * static_cast<Accum>(p0[z0]) +
                  w1 * static_cast<Accum>(p1[z0]));
              acc += a21[i2] * (
                  w0 * static_cast<Accum>(p0[z1]) +
                  w1 * static_cast<Accum>(p1[z1]));
              acc += a22[i2] * (
                  w0 * static_cast<Accum>(p0[z2]) +
                  w1 * static_cast<Accum>(p1[z2]));
              break;
            }
            case 2: {
              const int z0 = s20[i2];
              const int z1 = s21[i2];
              acc = a20[i2] * (
                  w0 * static_cast<Accum>(p0[z0]) +
                  w1 * static_cast<Accum>(p1[z0]));
              acc += a21[i2] * (
                  w0 * static_cast<Accum>(p0[z1]) +
                  w1 * static_cast<Accum>(p1[z1]));
              break;
            }
            case 1: {
              const int z0 = s20[i2];
              acc = a20[i2] * (
                  w0 * static_cast<Accum>(p0[z0]) +
                  w1 * static_cast<Accum>(p1[z0]));
              break;
            }
            default:
              break;
          }
          dst[i2] = static_cast<Scalar>(acc);
        }
        continue;
      }

      if (n0 == 2 && n1 == 1) {
        const Scalar* LS_RESTRICT p0 =
            in + static_cast<int64_t>(src0[0]) * in_s0 +
            static_cast<int64_t>(src1[0]) * in_s1;
        const Scalar* LS_RESTRICT p1 =
            in + static_cast<int64_t>(src0[1]) * in_s0 +
            static_cast<int64_t>(src1[0]) * in_s1;
        const Accum w0 = wt0[0] * wt1[0];
        const Accum w1 = wt0[1] * wt1[0];

        for (int64_t o2 = 0; o2 < out_n2; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          Accum acc = Accum(0);
          switch (c2[i2]) {
            case 3: {
              const int z0 = s20[i2];
              const int z1 = s21[i2];
              const int z2 = s22[i2];
              acc = a20[i2] * (
                  w0 * static_cast<Accum>(p0[z0]) +
                  w1 * static_cast<Accum>(p1[z0]));
              acc += a21[i2] * (
                  w0 * static_cast<Accum>(p0[z1]) +
                  w1 * static_cast<Accum>(p1[z1]));
              acc += a22[i2] * (
                  w0 * static_cast<Accum>(p0[z2]) +
                  w1 * static_cast<Accum>(p1[z2]));
              break;
            }
            case 2: {
              const int z0 = s20[i2];
              const int z1 = s21[i2];
              acc = a20[i2] * (
                  w0 * static_cast<Accum>(p0[z0]) +
                  w1 * static_cast<Accum>(p1[z0]));
              acc += a21[i2] * (
                  w0 * static_cast<Accum>(p0[z1]) +
                  w1 * static_cast<Accum>(p1[z1]));
              break;
            }
            case 1: {
              const int z0 = s20[i2];
              acc = a20[i2] * (
                  w0 * static_cast<Accum>(p0[z0]) +
                  w1 * static_cast<Accum>(p1[z0]));
              break;
            }
            default:
              break;
          }
          dst[i2] = static_cast<Scalar>(acc);
        }
        continue;
      }

      for (int64_t o2 = 0; o2 < out_n2; ++o2) {
        const size_t i2 = static_cast<size_t>(o2);
        const int n2 = static_cast<int>(c2[i2]);
        src2[0] = s20[i2];
        src2[1] = s21[i2];
        src2[2] = s22[i2];
        wt2[0] = a20[i2];
        wt2[1] = a21[i2];
        wt2[2] = a22[i2];

        Accum acc = Accum(0);
        for (int j0 = 0; j0 < n0; ++j0) {
          const int64_t base0 = static_cast<int64_t>(src0[j0]) * in_s0;
          const Accum wj0 = wt0[j0];
          for (int j1 = 0; j1 < n1; ++j1) {
            const int64_t base1 =
                base0 + static_cast<int64_t>(src1[j1]) * in_s1;
            const Accum wj01 = wj0 * wt1[j1];
            for (int j2 = 0; j2 < n2; ++j2) {
              acc += wj01 * wt2[j2] * static_cast<Accum>(
                  in[base1 + static_cast<int64_t>(src2[j2])]);
            }
          }
        }
        dst[static_cast<size_t>(o2)] = static_cast<Scalar>(acc);
      }
    }
  };

  run_parallel_for_shape(out_n0 * out_n1, plan2, in_shape, worker);
}

template <typename Scalar, typename Accum>
static inline Accum eval_axis2_weighted_rows_generic(
  const Scalar* LS_RESTRICT row0,
  const Scalar* LS_RESTRICT row1,
  const Scalar* LS_RESTRICT row2,
  int nrow,
  Accum wr0,
  Accum wr1,
  Accum wr2,
  const unsigned char* LS_RESTRICT c2,
  const int* LS_RESTRICT s20,
  const int* LS_RESTRICT s21,
  const int* LS_RESTRICT s22,
  const Accum* LS_RESTRICT a20,
  const Accum* LS_RESTRICT a21,
  const Accum* LS_RESTRICT a22,
  size_t i2)
{
  auto row_value = [&](int z) -> Accum {
    Accum value = wr0 * static_cast<Accum>(row0[static_cast<size_t>(z)]);
    if (nrow >= 2) {
      value += wr1 * static_cast<Accum>(row1[static_cast<size_t>(z)]);
    }
    if (nrow >= 3) {
      value += wr2 * static_cast<Accum>(row2[static_cast<size_t>(z)]);
    }
    return value;
  };

  switch (c2[i2]) {
    case 3:
      return a20[i2] * row_value(s20[i2]) +
             a21[i2] * row_value(s21[i2]) +
             a22[i2] * row_value(s22[i2]);
    case 2:
      return a20[i2] * row_value(s20[i2]) +
             a21[i2] * row_value(s21[i2]);
    case 1:
      return a20[i2] * row_value(s20[i2]);
    default:
      return Accum(0);
  }
}

template <typename Scalar>
static void resize_3d_linear_axis02_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p2,
  const Plan1D* preplanned0 = nullptr,
  const Plan1D* preplanned2 = nullptr)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::Fused3DLinearTotal);

  const int64_t in_n0 = in_shape[0];
  const int64_t in_n1 = in_shape[1];
  const int64_t in_n2 = in_shape[2];
  const int64_t out_n0 = out_shape[0];
  const int64_t out_n1 = out_shape[1];
  const int64_t out_n2 = out_shape[2];
  const int64_t in_s0 = in_n1 * in_n2;
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;

  if (out_n1 != in_n1) {
    LSParams p1 = p0;
    p1.zoom = 1.0;
    resize_3d_linear_t(in, out, in_shape, out_shape, p0, p1, p2);
    return;
  }

  std::shared_ptr<const Plan1D> plan0_handle;
  std::shared_ptr<const Plan1D> plan2_handle;
  if (preplanned0 == nullptr) {
    plan0_handle = get_plan_1d_cached(static_cast<int>(in_n0), p0);
    preplanned0 = plan0_handle.get();
  }
  if (preplanned2 == nullptr) {
    plan2_handle = get_plan_1d_cached(static_cast<int>(in_n2), p2);
    preplanned2 = plan2_handle.get();
  }
  const Plan1D& plan0 = *preplanned0;
  const Plan1D& plan2 = *preplanned2;
  const auto direct0 = direct_linear_plan_view<Accum>(plan0);
  const auto direct2 = direct_linear_plan_view<Accum>(plan2);

  if (!direct0.ok || !direct2.ok) {
    std::vector<int64_t> shape1 = {out_n0, in_n1, in_n2};
    std::vector<Scalar> tmp(
        static_cast<size_t>(out_n0) *
        static_cast<size_t>(in_n1) *
        static_cast<size_t>(in_n2));
    resize_along_axis_t(in, tmp.data(), in_shape, shape1, 0, p0, &plan0);
    resize_along_axis_t(
        tmp.data(), out, shape1, out_shape, 2, p2, &plan2);
    return;
  }

  const unsigned char* LS_RESTRICT c0 = direct0.count;
  const int* LS_RESTRICT s00 = direct0.src0;
  const int* LS_RESTRICT s01 = direct0.src1;
  const int* LS_RESTRICT s02 = direct0.src2;
  const Accum* LS_RESTRICT a00 = direct0.w0;
  const Accum* LS_RESTRICT a01 = direct0.w1;
  const Accum* LS_RESTRICT a02 = direct0.w2;
  const unsigned char* LS_RESTRICT c2 = direct2.count;
  const int* LS_RESTRICT s20 = direct2.src0;
  const int* LS_RESTRICT s21 = direct2.src1;
  const int* LS_RESTRICT s22 = direct2.src2;
  const Accum* LS_RESTRICT a20 = direct2.w0;
  const Accum* LS_RESTRICT a21 = direct2.w1;
  const Accum* LS_RESTRICT a22 = direct2.w2;
  const auto count2_run = longest_count_run(c2, out_n2, 2);
  const int64_t count2_begin = count2_run.first;
  const int64_t count2_end = count2_run.second;

  auto worker = [&](int64_t start, int64_t end) {
    for (int64_t line = start; line < end; ++line) {
      const int64_t o0 = line / in_n1;
      const int64_t i1 = line - o0 * in_n1;
      const size_t i0 = static_cast<size_t>(o0);
      const int n0 = static_cast<int>(c0[i0]);
      const Scalar* LS_RESTRICT row0 =
          in + static_cast<int64_t>(s00[i0]) * in_s0 + i1 * in_n2;
      const Scalar* LS_RESTRICT row1 =
          in + static_cast<int64_t>(s01[i0]) * in_s0 + i1 * in_n2;
      const Scalar* LS_RESTRICT row2 =
          in + static_cast<int64_t>(s02[i0]) * in_s0 + i1 * in_n2;
      Scalar* LS_RESTRICT dst = out + line * out_n2;
      const Accum wr0 = a00[i0];
      const Accum wr1 = a01[i0];
      const Accum wr2 = a02[i0];

      auto generic_range = [&](int64_t begin, int64_t finish) {
        for (int64_t o2 = begin; o2 < finish; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          dst[i2] = static_cast<Scalar>(
              eval_axis2_weighted_rows_generic(
                  row0, row1, row2, n0, wr0, wr1, wr2,
                  c2, s20, s21, s22, a20, a21, a22, i2));
        }
      };

      generic_range(0, count2_begin);
      if (n0 == 2) {
        for (int64_t o2 = count2_begin; o2 < count2_end; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          const int z0 = s20[i2];
          const int z1 = s21[i2];
          const Accum h0 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z0)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z0)]);
          const Accum h1 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z1)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z1)]);
          dst[i2] = static_cast<Scalar>(a20[i2] * h0 + a21[i2] * h1);
        }
      } else if (n0 == 1) {
        for (int64_t o2 = count2_begin; o2 < count2_end; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          const int z0 = s20[i2];
          const int z1 = s21[i2];
          dst[i2] = static_cast<Scalar>(
              wr0 * (
                  a20[i2] * static_cast<Accum>(row0[static_cast<size_t>(z0)]) +
                  a21[i2] * static_cast<Accum>(row0[static_cast<size_t>(z1)])));
        }
      } else if (n0 == 3) {
        for (int64_t o2 = count2_begin; o2 < count2_end; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          const int z0 = s20[i2];
          const int z1 = s21[i2];
          const Accum h0 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z0)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z0)]) +
              wr2 * static_cast<Accum>(row2[static_cast<size_t>(z0)]);
          const Accum h1 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z1)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z1)]) +
              wr2 * static_cast<Accum>(row2[static_cast<size_t>(z1)]);
          dst[i2] = static_cast<Scalar>(a20[i2] * h0 + a21[i2] * h1);
        }
      } else {
        generic_range(count2_begin, count2_end);
      }
      generic_range(count2_end, out_n2);
    }
  };

  run_parallel_for_shape(out_n0 * in_n1, plan2, in_shape, worker);
}

template <typename Scalar>
static void resize_3d_linear_axis12_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p1,
  const LSParams& p2,
  const Plan1D* preplanned1 = nullptr,
  const Plan1D* preplanned2 = nullptr)
{
  LSRESIZE_PROFILE_SCOPE(profile::Phase::Fused3DLinearTotal);

  const int64_t in_n0 = in_shape[0];
  const int64_t in_n1 = in_shape[1];
  const int64_t in_n2 = in_shape[2];
  const int64_t out_n0 = out_shape[0];
  const int64_t out_n1 = out_shape[1];
  const int64_t out_n2 = out_shape[2];
  const int64_t in_s0 = in_n1 * in_n2;
  const int64_t in_s1 = in_n2;
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;

  if (out_n0 != in_n0) {
    LSParams p0 = p1;
    p0.zoom = 1.0;
    resize_3d_linear_t(in, out, in_shape, out_shape, p0, p1, p2);
    return;
  }

  std::shared_ptr<const Plan1D> plan1_handle;
  std::shared_ptr<const Plan1D> plan2_handle;
  if (preplanned1 == nullptr) {
    plan1_handle = get_plan_1d_cached(static_cast<int>(in_n1), p1);
    preplanned1 = plan1_handle.get();
  }
  if (preplanned2 == nullptr) {
    plan2_handle = get_plan_1d_cached(static_cast<int>(in_n2), p2);
    preplanned2 = plan2_handle.get();
  }
  const Plan1D& plan1 = *preplanned1;
  const Plan1D& plan2 = *preplanned2;
  const auto direct1 = direct_linear_plan_view<Accum>(plan1);
  const auto direct2 = direct_linear_plan_view<Accum>(plan2);

  if (!direct1.ok || !direct2.ok) {
    std::vector<int64_t> shape1 = {in_n0, out_n1, in_n2};
    std::vector<Scalar> tmp(
        static_cast<size_t>(in_n0) *
        static_cast<size_t>(out_n1) *
        static_cast<size_t>(in_n2));
    resize_along_axis_t(in, tmp.data(), in_shape, shape1, 1, p1, &plan1);
    resize_along_axis_t(
        tmp.data(), out, shape1, out_shape, 2, p2, &plan2);
    return;
  }

  const unsigned char* LS_RESTRICT c1 = direct1.count;
  const int* LS_RESTRICT s10 = direct1.src0;
  const int* LS_RESTRICT s11 = direct1.src1;
  const int* LS_RESTRICT s12 = direct1.src2;
  const Accum* LS_RESTRICT a10 = direct1.w0;
  const Accum* LS_RESTRICT a11 = direct1.w1;
  const Accum* LS_RESTRICT a12 = direct1.w2;
  const unsigned char* LS_RESTRICT c2 = direct2.count;
  const int* LS_RESTRICT s20 = direct2.src0;
  const int* LS_RESTRICT s21 = direct2.src1;
  const int* LS_RESTRICT s22 = direct2.src2;
  const Accum* LS_RESTRICT a20 = direct2.w0;
  const Accum* LS_RESTRICT a21 = direct2.w1;
  const Accum* LS_RESTRICT a22 = direct2.w2;
  const auto count2_run = longest_count_run(c2, out_n2, 2);
  const int64_t count2_begin = count2_run.first;
  const int64_t count2_end = count2_run.second;

  auto worker = [&](int64_t start, int64_t end) {
    for (int64_t line = start; line < end; ++line) {
      const int64_t i0 = line / out_n1;
      const int64_t o1 = line - i0 * out_n1;
      const size_t i1 = static_cast<size_t>(o1);
      const int n1 = static_cast<int>(c1[i1]);
      const int64_t plane = i0 * in_s0;
      const Scalar* LS_RESTRICT row0 =
          in + plane + static_cast<int64_t>(s10[i1]) * in_s1;
      const Scalar* LS_RESTRICT row1 =
          in + plane + static_cast<int64_t>(s11[i1]) * in_s1;
      const Scalar* LS_RESTRICT row2 =
          in + plane + static_cast<int64_t>(s12[i1]) * in_s1;
      Scalar* LS_RESTRICT dst = out + line * out_n2;
      const Accum wr0 = a10[i1];
      const Accum wr1 = a11[i1];
      const Accum wr2 = a12[i1];

      auto generic_range = [&](int64_t begin, int64_t finish) {
        for (int64_t o2 = begin; o2 < finish; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          dst[i2] = static_cast<Scalar>(
              eval_axis2_weighted_rows_generic(
                  row0, row1, row2, n1, wr0, wr1, wr2,
                  c2, s20, s21, s22, a20, a21, a22, i2));
        }
      };

      generic_range(0, count2_begin);
      if (n1 == 2) {
        for (int64_t o2 = count2_begin; o2 < count2_end; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          const int z0 = s20[i2];
          const int z1 = s21[i2];
          const Accum h0 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z0)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z0)]);
          const Accum h1 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z1)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z1)]);
          dst[i2] = static_cast<Scalar>(a20[i2] * h0 + a21[i2] * h1);
        }
      } else if (n1 == 1) {
        for (int64_t o2 = count2_begin; o2 < count2_end; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          const int z0 = s20[i2];
          const int z1 = s21[i2];
          dst[i2] = static_cast<Scalar>(
              wr0 * (
                  a20[i2] * static_cast<Accum>(row0[static_cast<size_t>(z0)]) +
                  a21[i2] * static_cast<Accum>(row0[static_cast<size_t>(z1)])));
        }
      } else if (n1 == 3) {
        for (int64_t o2 = count2_begin; o2 < count2_end; ++o2) {
          const size_t i2 = static_cast<size_t>(o2);
          const int z0 = s20[i2];
          const int z1 = s21[i2];
          const Accum h0 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z0)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z0)]) +
              wr2 * static_cast<Accum>(row2[static_cast<size_t>(z0)]);
          const Accum h1 =
              wr0 * static_cast<Accum>(row0[static_cast<size_t>(z1)]) +
              wr1 * static_cast<Accum>(row1[static_cast<size_t>(z1)]) +
              wr2 * static_cast<Accum>(row2[static_cast<size_t>(z1)]);
          dst[i2] = static_cast<Scalar>(a20[i2] * h0 + a21[i2] * h1);
        }
      } else {
        generic_range(count2_begin, count2_end);
      }
      generic_range(count2_end, out_n2);
    }
  };

  run_parallel_for_shape(in_n0 * out_n1, plan2, in_shape, worker);
}

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------

void resize_along_axis(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p)
{
  resize_along_axis_t<double>(in, out, in_shape, out_shape, axis, p);
}

void resize_along_axis_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p)
{
  resize_along_axis_t<float>(in, out, in_shape, out_shape, axis, p);
}

void resize_along_axis_preplanned(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p,
  const Plan1D& plan)
{
  resize_along_axis_t<double>(
      in, out, in_shape, out_shape, axis, p, &plan);
}

void resize_along_axis_preplanned_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p,
  const Plan1D& plan)
{
  resize_along_axis_t<float>(
      in, out, in_shape, out_shape, axis, p, &plan);
}

void resize_2d_linear(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1)
{
  resize_2d_linear_t<double>(in, out, in_shape, out_shape, p0, p1);
}

void resize_2d_linear_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1)
{
  resize_2d_linear_t<float>(in, out, in_shape, out_shape, p0, p1);
}

void resize_2d_linear_preplanned(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const Plan1D& plan0,
  const Plan1D& plan1)
{
  resize_2d_linear_t<double>(
      in, out, in_shape, out_shape, p0, p1, &plan0, &plan1);
}

void resize_2d_linear_preplanned_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const Plan1D& plan0,
  const Plan1D& plan1)
{
  resize_2d_linear_t<float>(
      in, out, in_shape, out_shape, p0, p1, &plan0, &plan1);
}

void resize_3d_linear(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const LSParams& p2)
{
  resize_3d_linear_t<double>(in, out, in_shape, out_shape, p0, p1, p2);
}

void resize_3d_linear_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const LSParams& p2)
{
  resize_3d_linear_t<float>(in, out, in_shape, out_shape, p0, p1, p2);
}

void resize_3d_linear_preplanned(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const LSParams& p2,
  const Plan1D& plan0,
  const Plan1D& plan1,
  const Plan1D& plan2)
{
  resize_3d_linear_t<double>(
      in, out, in_shape, out_shape, p0, p1, p2,
      &plan0, &plan1, &plan2);
}

void resize_3d_linear_preplanned_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p1,
  const LSParams& p2,
  const Plan1D& plan0,
  const Plan1D& plan1,
  const Plan1D& plan2)
{
  resize_3d_linear_t<float>(
      in, out, in_shape, out_shape, p0, p1, p2,
      &plan0, &plan1, &plan2);
}

void resize_3d_linear_axis02(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p2)
{
  resize_3d_linear_axis02_t<double>(in, out, in_shape, out_shape, p0, p2);
}

void resize_3d_linear_axis02_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p2)
{
  resize_3d_linear_axis02_t<float>(in, out, in_shape, out_shape, p0, p2);
}

void resize_3d_linear_axis02_preplanned(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p2,
  const Plan1D& plan0,
  const Plan1D& plan2)
{
  resize_3d_linear_axis02_t<double>(
      in, out, in_shape, out_shape, p0, p2, &plan0, &plan2);
}

void resize_3d_linear_axis02_preplanned_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p0,
  const LSParams& p2,
  const Plan1D& plan0,
  const Plan1D& plan2)
{
  resize_3d_linear_axis02_t<float>(
      in, out, in_shape, out_shape, p0, p2, &plan0, &plan2);
}

void resize_3d_linear_axis12(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p1,
  const LSParams& p2)
{
  resize_3d_linear_axis12_t<double>(in, out, in_shape, out_shape, p1, p2);
}

void resize_3d_linear_axis12_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p1,
  const LSParams& p2)
{
  resize_3d_linear_axis12_t<float>(in, out, in_shape, out_shape, p1, p2);
}

void resize_3d_linear_axis12_preplanned(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p1,
  const LSParams& p2,
  const Plan1D& plan1,
  const Plan1D& plan2)
{
  resize_3d_linear_axis12_t<double>(
      in, out, in_shape, out_shape, p1, p2, &plan1, &plan2);
}

void resize_3d_linear_axis12_preplanned_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const LSParams& p1,
  const LSParams& p2,
  const Plan1D& plan1,
  const Plan1D& plan2)
{
  resize_3d_linear_axis12_t<float>(
      in, out, in_shape, out_shape, p1, p2, &plan1, &plan2);
}

} // namespace lsresize
