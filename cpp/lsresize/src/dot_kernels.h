// splineops/cpp/lsresize/src/dot_kernels.h
#pragma once

#include <cstdlib>  // std::getenv

#if defined(__AVX2__) || defined(__AVX512F__)
  #include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
#endif

namespace lsresize {

// -----------------------------------------------------------------------------
// AVX/FMA dot kernel with selective AVX-512 usage.
// - AVX-512 is used only when M is reasonably large (default: M >= 64) to
//   avoid frequency throttling penalties on some CPUs.
// - Force AVX2 via env: LSRESIZE_FORCE_AVX2=1
// -----------------------------------------------------------------------------
inline bool force_avx2() {
  const char* e = std::getenv("LSRESIZE_FORCE_AVX2");
  return (e && e[0] == '1');
}

#if defined(__AVX2__)
inline double hsum256(__m256d v) {
  __m128d vlow  = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1);
  vlow  = _mm_add_pd(vlow, vhigh);
  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  vlow  = _mm_add_sd(vlow, high64);
  return _mm_cvtsd_f64(vlow);
}
#endif

#if defined(__AVX512F__)
inline double hsum512(__m512d v) {
  // MSVC/Clang/GCC support this reduce add for AVX-512F.
  return _mm512_reduce_add_pd(v);
}
#endif

// Small-M unrolled dot for very short kernels (M <= 8)
// This is hit all the time for spline degrees 1..3.
inline double dot_small_unrolled(
  const double* w, 
  const double* v, 
  int M) 
{
  switch (M) {
    case 0:
      return 0.0;
    case 1:
      return w[0] * v[0];
    case 2:
      return w[0] * v[0] +
             w[1] * v[1];
    case 3:
      return w[0] * v[0] +
             w[1] * v[1] +
             w[2] * v[2];
    case 4:
      return w[0] * v[0] +
             w[1] * v[1] +
             w[2] * v[2] +
             w[3] * v[3];
    case 5:
      return w[0] * v[0] +
             w[1] * v[1] +
             w[2] * v[2] +
             w[3] * v[3] +
             w[4] * v[4];
    case 6:
      return w[0] * v[0] +
             w[1] * v[1] +
             w[2] * v[2] +
             w[3] * v[3] +
             w[4] * v[4] +
             w[5] * v[5];
    case 7:
      return w[0] * v[0] +
             w[1] * v[1] +
             w[2] * v[2] +
             w[3] * v[3] +
             w[4] * v[4] +
             w[5] * v[5] +
             w[6] * v[6];
    case 8:
      return w[0] * v[0] +
             w[1] * v[1] +
             w[2] * v[2] +
             w[3] * v[3] +
             w[4] * v[4] +
             w[5] * v[5] +
             w[6] * v[6] +
             w[7] * v[7];
    default:
      break;
  }

  // Fallback if someone calls with M > 8 (should be guarded by caller).
  double acc = 0.0;
  for (int t = 0; t < M; ++t) {
    acc += w[t] * v[t];
  }
  return acc;
}

// General dot product with optional AVX/NEON acceleration.
// Uses unrolled scalar path for M <= 8.
inline double dot_small(
  const double* w, 
  const double* v, 
  int M) 
{
  // Fast path for the common small-kernel cases (cubic/LS/oblique)
  if (M <= 8) {
    return dot_small_unrolled(w, v, M);
  }

#if defined(__AVX512F__)
  if (!force_avx2() && M >= 64) {
    __m512d acc0 = _mm512_setzero_pd();
    int t = 0;
    for (; t + 8 <= M; t += 8) {                // step by 8
      __m512d ww = _mm512_loadu_pd(w + t);
      __m512d vv = _mm512_loadu_pd(v + t);
      acc0 = _mm512_fmadd_pd(ww, vv, acc0);
    }
    double acc = hsum512(acc0);
    for (; t < M; ++t) {
      acc += w[t] * v[t];
    }
    return acc;
  }
#elif defined(__AVX2__)
  {
    __m256d acc0 = _mm256_setzero_pd();
    int t = 0;
    for (; t + 4 <= M; t += 4) {                // step by 4
      __m256d ww = _mm256_loadu_pd(w + t);
      __m256d vv = _mm256_loadu_pd(v + t);
      acc0 = _mm256_fmadd_pd(ww, vv, acc0);
    }
    double acc = hsum256(acc0);
    for (; t < M; ++t) {
      acc += w[t] * v[t];
    }
    return acc;
  }
#elif defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  {
    float64x2_t acc0 = vdupq_n_f64(0.0);
    int t = 0;
    for (; t + 2 <= M; t += 2) {
      float64x2_t ww = vld1q_f64(w + t);
      float64x2_t vv = vld1q_f64(v + t);
      acc0 = vmlaq_f64(acc0, ww, vv);  // acc0 += ww * vv
    }
    double tmp[2];
    vst1q_f64(tmp, acc0);
    double acc = tmp[0] + tmp[1];
    for (; t < M; ++t) {
      acc += w[t] * v[t];
    }
    return acc;
  }
#endif

  // Scalar fallback for mid-sized kernels
  double acc = 0.0;
  for (int t = 0; t < M; ++t) {
    acc += w[t] * v[t];
  }
  return acc;
}

} // namespace lsresize