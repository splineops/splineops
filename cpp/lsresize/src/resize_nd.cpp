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
        accumulate_row_runs_colmajor_f32_set(
            coeff.data(),
            Bs,
            B,
            plan,
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
            accumulate_row_colmajor_f32_set(
                coeff.data(),
                Bs,
                B,
                plan,
                weights,
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

      accumulate_row_runs_colmajor_f32_set(
          coeff.data(),
          Bs,
          B,
          plan,
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

  if (should_use_batched_axis(
          batched_axis_mode(),
          in_shape,
          p,
          plan,
          nlines)) {
    if constexpr (std::is_same_v<Scalar, float>) {
      if (float32_internal_enabled()) {
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

} // namespace lsresize
