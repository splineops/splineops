// splineops/cpp/lsresize/src/resize_nd.cpp
#include "resize_nd.h"
#include "utils.h"
#include "parallel_utils.h"
#include "resize_1d.h"
#include "filters.h"

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

static inline int env_int_or_default(const char* name, int fallback)
{
  if (const char* value = std::getenv(name)) {
    if (int parsed = std::atoi(value); parsed > 0) {
      return parsed;
    }
  }
  return fallback;
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
  return in_shape.size() == 2 &&
         p.analy_degree < 0 &&
         p.synthe_degree == p.interp_degree &&
         (p.interp_degree == 2 || p.interp_degree == 3);
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

static inline int specialized_preset_max_support(const LSParams& p)
{
  if (p.analy_degree < 0 && p.synthe_degree == p.interp_degree) {
    if (p.interp_degree == 1) return 3;  // linear
    if (p.interp_degree == 3) return 5;  // cubic
    return 0;
  }

  if (p.interp_degree == 1 &&
      p.analy_degree == 0 &&
      p.synthe_degree == 1) {
    return 4;  // linear-antialiasing: total degree 2
  }

  if (p.interp_degree == 3 &&
      p.analy_degree == 1 &&
      p.synthe_degree == 3) {
    return 7;  // cubic-antialiasing: total degree 5
  }

  return 0;
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

  // Conservative default: local benchmarks show stable wins for large 2-D
  // passes, while 3-D cases are still mixed.
  if (in_shape.size() != 2) {
    return false;
  }
  if (p.interp_degree <= 0) {
    return false;
  }
  if (nlines < 128) {
    return false;
  }
  if (plan.N < 64 || plan.out_total < 64) {
    return false;
  }
  return true;
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
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

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
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

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
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

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
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

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
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

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
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();
  for (int t = begin; t < endw; ++t) {
    const size_t ti = static_cast<size_t>(t);
    const int src = coeff_src[ti];
    acc += weights[ti] * coeff_sgn[ti] *
           static_cast<double>(line[static_cast<int64_t>(src) * stride]);
  }
  return acc;
}

template <typename Weight>
static inline bool build_direct_linear_plan(
  const Plan1D& plan,
  std::vector<unsigned char>& count,
  std::vector<int>& src0,
  std::vector<int>& src1,
  std::vector<int>& src2,
  std::vector<Weight>& w0,
  std::vector<Weight>& w1,
  std::vector<Weight>& w2)
{
  const int outN = plan.outN;
  count.assign(static_cast<size_t>(outN), 0);
  src0.assign(static_cast<size_t>(outN), 0);
  src1.assign(static_cast<size_t>(outN), 0);
  src2.assign(static_cast<size_t>(outN), 0);
  w0.assign(static_cast<size_t>(outN), 0.0);
  w1.assign(static_cast<size_t>(outN), 0.0);
  w2.assign(static_cast<size_t>(outN), 0.0);

  const double* LS_RESTRICT weights = plan.weights.data();
  const int* LS_RESTRICT coeff_src = plan.coeff_src.data();
  const double* LS_RESTRICT coeff_sgn = plan.coeff_sgn.data();

  for (int l = 0; l < outN; ++l) {
    const size_t li = static_cast<size_t>(l);
    const int begin = plan.row_ptr[li];
    const int endw = plan.row_ptr[li + 1];
    const int m = endw - begin;
    if (m < 0 || m > 3) {
      return false;
    }

    count[li] = static_cast<unsigned char>(m);
    const int k0 = plan.kmin[li];
    const int kmax = k0 + m - 1;
    const bool interior = (m == 0 || (k0 >= 0 && kmax < plan.N));

    int src[3] = {0, 0, 0};
    double ww[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < m; ++j) {
      const int t = begin + j;
      const size_t ti = static_cast<size_t>(t);
      if (interior) {
        src[j] = k0 + j;
        ww[j] = weights[ti];
      } else {
        src[j] = coeff_src[ti];
        ww[j] = weights[ti] * coeff_sgn[ti];
      }
    }

    src0[li] = src[0];
    src1[li] = src[1];
    src2[li] = src[2];
    w0[li] = static_cast<Weight>(ww[0]);
    w1[li] = static_cast<Weight>(ww[1]);
    w2[li] = static_cast<Weight>(ww[2]);
  }

  return true;
}

static inline std::pair<int64_t, int64_t> longest_count_run(
  const std::vector<unsigned char>& count,
  unsigned char target)
{
  int64_t best_begin = 0;
  int64_t best_end = 0;
  int64_t run_begin = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(count.size()); ++i) {
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
  const int64_t n = static_cast<int64_t>(count.size());
  if (run_begin >= 0 && n - run_begin > best_end - best_begin) {
    best_begin = run_begin;
    best_end = n;
  }
  return {best_begin, best_end};
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
  const int64_t in_w = in_shape[1];
  const int64_t out_w = out_shape[1];
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;
  std::vector<unsigned char> count;
  std::vector<int> src0;
  std::vector<int> src1;
  std::vector<int> src2;
  std::vector<Accum> w0;
  std::vector<Accum> w1;
  std::vector<Accum> w2;

  if (!build_direct_linear_plan(plan, count, src0, src1, src2, w0, w1, w2)) {
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

    run_parallel_or_serial(nlines, plan, fallback);
    return;
  }

  const unsigned char* LS_RESTRICT c = count.data();
  const int* LS_RESTRICT s0 = src0.data();
  const int* LS_RESTRICT s1 = src1.data();
  const int* LS_RESTRICT s2 = src2.data();
  const Accum* LS_RESTRICT a0 = w0.data();
  const Accum* LS_RESTRICT a1 = w1.data();
  const Accum* LS_RESTRICT a2 = w2.data();
  const auto count2_run = longest_count_run(count, 2);
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

  run_parallel_or_serial(nlines, plan, worker);
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
  const int D = static_cast<int>(in_shape.size());
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;
  std::vector<unsigned char> count;
  std::vector<int> src0;
  std::vector<int> src1;
  std::vector<int> src2;
  std::vector<Accum> w0;
  std::vector<Accum> w1;
  std::vector<Accum> w2;

  if (!build_direct_linear_plan(plan, count, src0, src1, src2, w0, w1, w2)) {
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

    run_parallel_or_serial(nlines, plan, fallback);
    return;
  }

  const unsigned char* LS_RESTRICT c = count.data();
  const int* LS_RESTRICT s0 = src0.data();
  const int* LS_RESTRICT s1 = src1.data();
  const int* LS_RESTRICT s2 = src2.data();
  const Accum* LS_RESTRICT a0 = w0.data();
  const Accum* LS_RESTRICT a1 = w1.data();
  const Accum* LS_RESTRICT a2 = w2.data();

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

  run_parallel_or_serial(nlines, plan, worker);
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
  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int batch_lines =
      std::max(1, env_int_or_default("LSRESIZE_BATCH_LINES", kDefaultBatchLines));
  const bool axis_contig_in = (in_strides[static_cast<size_t>(axis)] == 1);
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p)
                        : 0;

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

      for (int b = 0; b < B; ++b) {
        int64_t in_off = 0;
        int64_t out_off = 0;
        line_offsets(
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

        if (axis_contig_in) {
          const Scalar* src = in + in_off;
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                static_cast<double>(src[static_cast<size_t>(n)]);
          }
        } else {
          const int64_t stride = in_strides[static_cast<size_t>(axis)];
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                static_cast<double>(in[in_off + static_cast<int64_t>(n) * stride]);
          }
        }
      }

      get_interpolation_coefficients_colmajor(coeff, B, N, p.interp_degree);

      if (axis_contig_out) {
        y.resize(static_cast<size_t>(outN) * Bs);
        accumulate_row_runs_colmajor_preset_set(
            coeff.data(),
            Bs,
            B,
            plan,
            max_support,
            y.data());

        for (int b = 0; b < B; ++b) {
          const int64_t out_off = out_offsets[static_cast<size_t>(b)];
          Scalar* dst = out + out_off;
          for (int l = 0; l < outN; ++l) {
            dst[static_cast<size_t>(l)] =
                static_cast<Scalar>(y[static_cast<size_t>(l) * Bs +
                                      static_cast<size_t>(b)]);
          }
        }
      } else {
        accum.resize(Bs);
        const int64_t stride = out_strides[static_cast<size_t>(axis)];
        const double* weights = plan.weights.data();
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
      }
    }
  };

  run_parallel_or_serial(nlines, plan, worker);
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
  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int out_total = plan.out_total;
  const int corr_degree = (p.analy_degree < 0)
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);
  const int batch_lines =
      std::max(1, env_int_or_default("LSRESIZE_BATCH_LINES", kDefaultBatchLines));
  const bool axis_contig_in = (in_strides[static_cast<size_t>(axis)] == 1);
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p)
                        : 0;

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

      for (int b = 0; b < B; ++b) {
        int64_t in_off = 0;
        int64_t out_off = 0;
        line_offsets(
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

        if (axis_contig_in) {
          const Scalar* src = in + in_off;
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                static_cast<double>(src[static_cast<size_t>(n)]);
          }
        } else {
          const int64_t stride = in_strides[static_cast<size_t>(axis)];
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                static_cast<double>(in[in_off + static_cast<int64_t>(n) * stride]);
          }
        }
      }

      get_interpolation_coefficients_colmajor(coeff, B, N, p.interp_degree);
      if (p.analy_degree >= 0) {
        do_integ_colmajor(
            coeff,
            B,
            N,
            p.analy_degree + 1,
            average,
            filter_work);
      }

      accumulate_row_runs_colmajor_preset_set(
          coeff.data(),
          Bs,
          B,
          plan,
          max_support,
          y.data());

      if (p.analy_degree >= 0) {
        do_diff_colmajor(y, B, out_total, p.analy_degree + 1, filter_work);
        for (int l = 0; l < out_total; ++l) {
          double* y_col = y.data() + static_cast<size_t>(l) * Bs;
          for (int b = 0; b < B; ++b) {
            y_col[static_cast<size_t>(b)] += average[static_cast<size_t>(b)];
          }
        }
        get_interpolation_coefficients_colmajor(y, B, out_total, corr_degree);
        get_samples_colmajor(y, B, out_total, p.synthe_degree, filter_work);
      }

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
  };

  run_parallel_or_serial(nlines, plan, worker);
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
  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int batch_lines =
      std::max(1, env_int_or_default("LSRESIZE_BATCH_LINES", kDefaultBatchLines));
  const bool axis_contig_in = (in_strides[static_cast<size_t>(axis)] == 1);
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p)
                        : 0;

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

      for (int b = 0; b < B; ++b) {
        int64_t in_off = 0;
        int64_t out_off = 0;
        line_offsets(
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

        if (axis_contig_in) {
          const float* src = in + in_off;
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                src[static_cast<size_t>(n)];
          }
        } else {
          const int64_t stride = in_strides[static_cast<size_t>(axis)];
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                in[in_off + static_cast<int64_t>(n) * stride];
          }
        }
      }

      get_interpolation_coefficients_colmajor_f32(
          coeff, B, N, p.interp_degree);

      if (axis_contig_out) {
        y.resize(static_cast<size_t>(outN) * Bs);
        accumulate_row_runs_colmajor_f32_preset_set(
            coeff.data(),
            Bs,
            B,
            plan,
            max_support,
            y.data());

        for (int b = 0; b < B; ++b) {
          const int64_t out_off = out_offsets[static_cast<size_t>(b)];
          float* dst = out + out_off;
          for (int l = 0; l < outN; ++l) {
            dst[static_cast<size_t>(l)] =
                y[static_cast<size_t>(l) * Bs + static_cast<size_t>(b)];
          }
        }
      } else {
        accum.resize(Bs);
        const int64_t stride = out_strides[static_cast<size_t>(axis)];
        const double* weights = plan.weights.data();
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
  };

  run_parallel_or_serial(nlines, plan, worker);
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
  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int out_total = plan.out_total;
  const int corr_degree = (p.analy_degree < 0)
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);
  const int batch_lines =
      std::max(1, env_int_or_default("LSRESIZE_BATCH_LINES", kDefaultBatchLines));
  const bool axis_contig_in = (in_strides[static_cast<size_t>(axis)] == 1);
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);
  const int max_support = specialized_presets_enabled()
                        ? specialized_preset_max_support(p)
                        : 0;

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

      for (int b = 0; b < B; ++b) {
        int64_t in_off = 0;
        int64_t out_off = 0;
        line_offsets(
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

        if (axis_contig_in) {
          const float* src = in + in_off;
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                src[static_cast<size_t>(n)];
          }
        } else {
          const int64_t stride = in_strides[static_cast<size_t>(axis)];
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                in[in_off + static_cast<int64_t>(n) * stride];
          }
        }
      }

      // The projection pipeline is linear and should preserve constants. Run
      // it on a DC-centered residual so float32 recursive filters do not turn a
      // constant line into small boundary drift.
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

      get_interpolation_coefficients_colmajor_f32(
          coeff, B, N, p.interp_degree);
      if (p.analy_degree >= 0) {
        do_integ_colmajor_f32(
            coeff,
            B,
            N,
            p.analy_degree + 1,
            average,
            filter_work);
      }

      accumulate_row_runs_colmajor_f32_preset_set(
          coeff.data(),
          Bs,
          B,
          plan,
          max_support,
          y.data());

      if (p.analy_degree >= 0) {
        do_diff_colmajor_f32(y, B, out_total, p.analy_degree + 1, filter_work);
        for (int l = 0; l < out_total; ++l) {
          float* y_col = y.data() + static_cast<size_t>(l) * Bs;
          for (int b = 0; b < B; ++b) {
            y_col[static_cast<size_t>(b)] += average[static_cast<size_t>(b)];
          }
        }
        get_interpolation_coefficients_colmajor_f32(y, B, out_total, corr_degree);
        get_samples_colmajor_f32(y, B, out_total, p.synthe_degree, filter_work);
      }

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
  };

  run_parallel_or_serial(nlines, plan, worker);
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
  const LSParams& p)
{
  const int D = static_cast<int>(in_shape.size());
  const auto in_strides  = strides_from_shape(in_shape);
  const auto out_strides = strides_from_shape(out_shape);

  // Early identity short-circuit on this axis:
  {
    const double eps = 1e-12;
    const bool identity_axis =
        (out_shape[static_cast<size_t>(axis)] ==
         in_shape[static_cast<size_t>(axis)]) &&
        (std::abs(p.zoom - 1.0) <= eps) &&
        (p.analy_degree < 0); // Standard interpolation (no projection)
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

  // Build or reuse the per-axis plan ONCE (shared read-only across threads)
  const int N_line = static_cast<int>(in_shape[static_cast<size_t>(axis)]);
  const auto plan_handle = get_plan_1d_cached(N_line, p);
  const Plan1D& plan = *plan_handle;

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
      int64_t in_off  = 0;
      int64_t out_off = 0;
      for (int d = 0; d < D; ++d) {
        if (d != axis) {
          in_off  += idx[static_cast<size_t>(d)] *
                     in_strides[static_cast<size_t>(d)];
          out_off += idx[static_cast<size_t>(d)] *
                     out_strides[static_cast<size_t>(d)];
        }
      }

      // --- Fast path: axis contiguous in both in & out and Scalar == double ---
      if constexpr (std::is_same_v<Scalar, double>) {
        if (axis_contig_in && axis_contig_out) {
          // Direct 1-D resize on raw buffers, no gather/scatter via vectors.
          resize_1d_line_contiguous(in + in_off, out + out_off, p, plan, workspace);
          continue;
        }
      }

      // --- Fallback: gather into workspace.line (double), run 1-D core from line ---

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

      // Fast planned path with workspace reuse (double internal)
      resize_1d_line_buffered(workspace.line, line_out, p, plan, workspace);

      // Scatter to output (Scalar storage)
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
  };

  // Centralized scheduling: OpenMP, std::thread, or serial
  run_parallel_or_serial(nlines, plan, worker);
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
  const LSParams& p1)
{
  const int64_t in_h = in_shape[0];
  const int64_t in_w = in_shape[1];
  const int64_t out_h = out_shape[0];
  const int64_t out_w = out_shape[1];
  using Accum = std::conditional_t<std::is_same_v<Scalar, float>, float, double>;

  const auto row_plan_handle =
      get_plan_1d_cached(static_cast<int>(in_h), p0);
  const auto col_plan_handle =
      get_plan_1d_cached(static_cast<int>(in_w), p1);
  const Plan1D& row_plan = *row_plan_handle;
  const Plan1D& col_plan = *col_plan_handle;

  std::vector<unsigned char> row_count;
  std::vector<int> row_src0;
  std::vector<int> row_src1;
  std::vector<int> row_src2;
  std::vector<Accum> row_w0;
  std::vector<Accum> row_w1;
  std::vector<Accum> row_w2;
  std::vector<unsigned char> col_count;
  std::vector<int> col_src0;
  std::vector<int> col_src1;
  std::vector<int> col_src2;
  std::vector<Accum> col_w0;
  std::vector<Accum> col_w1;
  std::vector<Accum> col_w2;

  const bool have_row_plan = build_direct_linear_plan(
      row_plan,
      row_count,
      row_src0,
      row_src1,
      row_src2,
      row_w0,
      row_w1,
      row_w2);
  const bool have_col_plan = build_direct_linear_plan(
      col_plan,
      col_count,
      col_src0,
      col_src1,
      col_src2,
      col_w0,
      col_w1,
      col_w2);
  if (!have_row_plan || !have_col_plan) {
    std::vector<Scalar> tmp(
        static_cast<size_t>(out_h) * static_cast<size_t>(in_w));
    std::vector<int64_t> mid_shape = {out_h, in_w};
    resize_along_axis_2d_linear_direct(
        in, tmp.data(), in_shape, mid_shape, 0, row_plan, in_w);
    resize_along_axis_2d_linear_direct(
        tmp.data(), out, mid_shape, out_shape, 1, col_plan, out_h);
    return;
  }

  const unsigned char* LS_RESTRICT rc = row_count.data();
  const int* LS_RESTRICT rs0 = row_src0.data();
  const int* LS_RESTRICT rs1 = row_src1.data();
  const int* LS_RESTRICT rs2 = row_src2.data();
  const Accum* LS_RESTRICT rw0 = row_w0.data();
  const Accum* LS_RESTRICT rw1 = row_w1.data();
  const Accum* LS_RESTRICT rw2 = row_w2.data();
  const unsigned char* LS_RESTRICT cc = col_count.data();
  const int* LS_RESTRICT cs0 = col_src0.data();
  const int* LS_RESTRICT cs1 = col_src1.data();
  const int* LS_RESTRICT cs2 = col_src2.data();
  const Accum* LS_RESTRICT cw0 = col_w0.data();
  const Accum* LS_RESTRICT cw1 = col_w1.data();
  const Accum* LS_RESTRICT cw2 = col_w2.data();
  const auto col2_run = longest_count_run(col_count, 2);
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

  run_parallel_or_serial(out_h, row_plan, worker);
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

} // namespace lsresize
