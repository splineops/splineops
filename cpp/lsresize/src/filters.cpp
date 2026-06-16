// splineops/cpp/lsresize/src/filters.cpp
#include "filters.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace lsresize {

namespace {

bool env_equals_ci_local(const char* value, const char* expected)
{
  if (value == nullptr || expected == nullptr) {
    return false;
  }
  while (*value != '\0' && *expected != '\0') {
    char a = *value;
    char b = *expected;
    if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
    if (a != b) {
      return false;
    }
    ++value;
    ++expected;
  }
  return *value == '\0' && *expected == '\0';
}

bool env_flag_enabled_default_true_local(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return !(value[0] == '0' ||
           env_equals_ci_local(value, "off") ||
           env_equals_ci_local(value, "false") ||
           env_equals_ci_local(value, "no"));
}

bool rowwise_initial_causal_enabled()
{
  return env_flag_enabled_default_true_local("LSRESIZE_ROWWISE_INITIAL_CAUSAL");
}

} // namespace

double initial_causal(
  const std::vector<double>& c, 
  double z, 
  double tol) 
{
  const size_t N = c.size();
  if (N == 0) return 0.0;
  if (N == 1) return c[0];

  size_t horizon = N;
  if (tol > 0.0) {
    horizon = std::min(N, size_t(2 + std::log(tol) / std::log(std::abs(z))));
  }

  if (horizon < N) {
    double sum = c[0];
    double p = z;
    for (size_t n = 1; n < horizon; ++n) {
      sum += p * c[n];
      p *= z;
    }
    return sum;
  }

  const double zn = std::pow(z, double(N - 1));
  double sum = c[0] + zn * c[N - 1];
  // Exact finite-length mirror-boundary terms: z^n and z^(2N-2-n).
  double p1 = z;          // z^1
  double p2 = (zn * zn) / z;  // z^(2N-3)
  for (size_t n = 1; n + 1 < N; ++n) {
    sum += (p1 + p2) * c[n];
    p1 *= z;            // z^n -> z^(n+1)
    p2 /= z;            // z^(2N-2-n) -> z^(2N-3-n)
  }

  const double denom = 1.0 - (zn * zn);   // 1 - z^(2N-2)
  return sum / denom;
}

double initial_anti_causal(
  const std::vector<double>& c, 
  double z) 
{
  if (c.size() < 2) return 0.0;
  return (z * c[c.size()-2] + c.back()) * z / (z*z - 1.0);
}

void symmetric_fir(
  const std::vector<double>& h, 
  const std::vector<double>& c, 
  std::vector<double>& s) 
{
  const size_t N = c.size();
  if (s.size() != N) s.assign(N, 0.0);
  const size_t L = h.size();
  if (L == 0 || N == 0) return;

  if (L == 2) {
    if (N >= 2) {
      s[0]   = h[0]*c[0] + 2.0*h[1]*c[1];
      for (size_t i=1; i+1<N; ++i) s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]);
      s[N-1] = h[0]*c[N-1] + 2.0*h[1]*c[N-2];
    } else {
      s[0] = (h[0] + 2.0*h[1]) * c[0];
    }
    return;
  }

  if (L == 3) {
    if (N >= 4) {
      s[0]   = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2];
      s[1]   = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]);
      for (size_t i=2; i+2<N; ++i)
        s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]) + h[2]*(c[i-2]+c[i+2]);
      s[N-2] = h[0]*c[N-2] + h[1]*(c[N-3]+c[N-1]) + h[2]*(c[N-4]+c[N-2]);
      s[N-1] = h[0]*c[N-1] + 2.0*h[1]*c[N-2] + 2.0*h[2]*c[N-3];
    } else if (N == 3) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + 2.0*h[2]*c[1];
      s[2] = h[0]*c[2] + 2.0*h[1]*c[1] + 2.0*h[2]*c[0];
    } else if (N == 2) {
      s[0] = (h[0] + 2.0*h[2]) * c[0] + 2.0*h[1]*c[1];
      s[1] = (h[0] + 2.0*h[2]) * c[1] + 2.0*h[1]*c[0];
    } else { // N==1
      s[0] = (h[0] + 2.0*(h[1]+h[2])) * c[0];
    }
    return;
  }

  if (L == 4) {
    if (N >= 6) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4]);
      s[2] = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[4]) + h[3]*(c[1]+c[5]);
      for (size_t i=3; i+3<N; ++i)
        s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]) + h[2]*(c[i-2]+c[i+2]) + h[3]*(c[i-3]+c[i+3]);
      s[N-3] = h[0]*c[N-3] + h[1]*(c[N-4]+c[N-2]) + h[2]*(c[N-5]+c[N-1]) + h[3]*(c[N-6]+c[N-2]);
      s[N-2] = h[0]*c[N-2] + h[1]*(c[N-3]+c[N-1]) + h[2]*(c[N-4]+c[N-2]) + h[3]*(c[N-5]+c[N-3]);
      s[N-1] = h[0]*c[N-1] + 2.0*h[1]*c[N-2] + 2.0*h[2]*c[N-3] + 2.0*h[3]*c[N-4];
    } else if (N == 5) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4]);
      s[2] = h[0]*c[2] + (h[1]+h[3])*(c[1]+c[3]) + h[2]*(c[0]+c[4]);
      s[3] = h[0]*c[3] + h[1]*(c[2]+c[4]) + h[2]*(c[1]+c[3]) + h[3]*(c[0]+c[2]);
      s[4] = h[0]*c[4] + 2.0*h[1]*c[3] + 2.0*h[2]*c[2] + 2.0*h[3]*c[1];
    } else if (N == 4) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + 2.0*h[3]*c[2];
      s[2] = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[2]) + 2.0*h[3]*c[1];
      s[3] = h[0]*c[3] + 2.0*h[1]*c[2] + 2.0*h[2]*c[1] + 2.0*h[3]*c[0];
    } else if (N == 3) {
      s[0] = h[0]*c[0] + 2.0*(h[1]+h[3])*c[1] + 2.0*h[2]*c[2];
      s[1] = h[0]*c[1] + (h[1]+h[3])*(c[0]+c[2]) + 2.0*h[2]*c[1];
      s[2] = h[0]*c[2] + 2.0*(h[1]+h[3])*c[1] + 2.0*h[2]*c[0];
    } else if (N == 2) {
      s[0] = (h[0]+2.0*h[2])*c[0] + 2.0*(h[1]+h[3])*c[1];
      s[1] = (h[0]+2.0*h[2])*c[1] + 2.0*(h[1]+h[3])*c[0];
    } else { // N==1
      s[0] = (h[0] + 2.0*(h[1]+h[2]+h[3])) * c[0];
    }
    return;
  }

  throw std::invalid_argument("Invalid filter half-length (should be 2..4)");
}

double do_integ(
  std::vector<double>& c, 
  int nb) 
{
  const size_t N = c.size();
  if (N <= 1 || nb <= 0) {
    return 0.0;  // nothing to integrate; no average to restore
  }

  auto avg_of = [&](const std::vector<double>& x)->double {
    const double sum = std::accumulate(x.begin(), x.end(), 0.0);
    return (2.0*sum - x.back() - x.front()) / (2.0*N - 2.0);
  };

  double m = 0.0, average = 0.0;
  if (nb >= 1) {
    average = avg_of(c);
    integ_sa(c, average);
  }
  if (nb >= 2) {
    // was: std::vector<double> tmp = c; integ_as(tmp, c);
    integ_as(c, c);
  }
  if (nb >= 3) {
    m = avg_of(c);
    integ_sa(c, m);
  }
  if (nb >= 4) {
    // was: std::vector<double> tmp = c; integ_as(tmp, c);
    integ_as(c, c);
  }
  return average;
}

void integ_sa(
  std::vector<double>& c, 
  double m) 
{
  c[0] -= m; c[0] *= 0.5;
  for (size_t i = 1; i < c.size(); ++i) { c[i] -= m; c[i] += c[i-1]; }
}

void integ_as(
  const std::vector<double>& c, 
  std::vector<double>& y) 
{
  const size_t N = c.size();
  y.resize(N);
  if (N == 0) return;

  if (&c == &y) {
    // In-place variant: read & write the same vector
    double c0 = y[0];
    if (N > 1) {
      double accum = 0.0;
      for (size_t i = 1; i < N; ++i) {
        double tmp = y[i];                 // original c[i]
        y[i] = (i == 1) ? 0.0 : -accum;    // y[i] = 0 for i=1, otherwise -sum c[1..i-1]
        accum += tmp;                      // accum = sum c[1..i]
      }
    }
    y[0] = c0;
  } else {
    // Separate input/output buffers
    y[0] = c[0];
    if (N > 1) {
      double accum = 0.0;
      for (size_t i = 1; i < N; ++i) {
        double tmp = c[i];
        y[i] = (i == 1) ? 0.0 : -accum;
        accum += tmp;
      }
    }
  }
}

void do_diff(
  std::vector<double>& c, 
  int nb) 
{
  const size_t N = c.size();
  if (N == 0 || nb <= 0) return;
  if (nb == 1) { diff_as(c); return; }
  if (nb == 2) { diff_sa(c); diff_as(c); return; }
  if (nb == 3) { diff_as(c); diff_sa(c); diff_as(c); return; }
  if (nb >= 4) { diff_sa(c); diff_as(c); diff_sa(c); diff_as(c); return; }
}

void diff_sa(std::vector<double>& c) 
{
  if (c.size() < 2) return;
  double old = c[c.size()-2];
  for (size_t i = 0; i + 1 < c.size(); ++i) c[i] = c[i] - c[i+1];
  c.back() -= old;
}

void diff_as(std::vector<double>& c) 
{
  if (c.size() < 2) { if (!c.empty()) c[0] *= 2.0; return; }
  for (size_t i = c.size()-1; i > 0; --i) c[i] = c[i] - c[i-1];
  c[0] *= 2.0;
}

// -----------------------------------------------------------------------------
// Batched col-major helpers: C[n * B + b]
// -----------------------------------------------------------------------------

static double initial_causal_colmajor_scalar(
  const std::vector<double>& c,
  int B,
  int N,
  int b,
  double z,
  size_t horizon)
{
  if (N == 0) return 0.0;
  if (N == 1) return c[static_cast<size_t>(b)];

  const size_t Bs = static_cast<size_t>(B);
  const size_t bi = static_cast<size_t>(b);

  if (horizon < static_cast<size_t>(N)) {
    const double* ptr = c.data() + bi;
    double sum = *ptr;
    double p = z;
    for (size_t n = 1; n < horizon; ++n) {
      ptr += Bs;
      sum += p * (*ptr);
      p *= z;
    }
    return sum;
  }

  const double zn = std::pow(z, double(N - 1));
  const double* ptr = c.data() + bi;
  double sum = *ptr + zn * (*(ptr + static_cast<size_t>(N - 1) * Bs));
  double p1 = z;
  double p2 = (zn * zn) / z;
  for (int n = 1; n + 1 < N; ++n) {
    ptr += Bs;
    sum += (p1 + p2) * (*ptr);
    p1 *= z;
    p2 /= z;
  }

  return sum / (1.0 - (zn * zn));
}

static inline size_t initial_causal_horizon_colmajor(
  int N,
  double z,
  double tol = 1e-10)
{
  size_t horizon = static_cast<size_t>(std::max(N, 0));
  if (N > 0 && tol > 0.0) {
    horizon = std::min(
        static_cast<size_t>(N),
        static_cast<size_t>(2 + std::log(tol) / std::log(std::abs(z))));
  }
  return horizon;
}

static void apply_interpolation_pole_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  double z,
  size_t horizon)
{
  const size_t Bs = static_cast<size_t>(B);
  if (rowwise_initial_causal_enabled()) {
    double* first = c.data();
    if (horizon < static_cast<size_t>(N)) {
      double p = z;
      for (size_t n = 1; n < horizon; ++n) {
        const double* row = c.data() + n * Bs;
        for (int b = 0; b < B; ++b) {
          first[static_cast<size_t>(b)] += p * row[static_cast<size_t>(b)];
        }
        p *= z;
      }
    } else {
      const double zn = std::pow(z, double(N - 1));
      const double* last = c.data() + static_cast<size_t>(N - 1) * Bs;
      for (int b = 0; b < B; ++b) {
        first[static_cast<size_t>(b)] += zn * last[static_cast<size_t>(b)];
      }
      double p1 = z;
      double p2 = (zn * zn) / z;
      for (int n = 1; n + 1 < N; ++n) {
        const double* row = c.data() + static_cast<size_t>(n) * Bs;
        const double w = p1 + p2;
        for (int b = 0; b < B; ++b) {
          first[static_cast<size_t>(b)] += w * row[static_cast<size_t>(b)];
        }
        p1 *= z;
        p2 /= z;
      }
      const double denom = 1.0 - (zn * zn);
      for (int b = 0; b < B; ++b) {
        first[static_cast<size_t>(b)] /= denom;
      }
    }
  } else {
    for (int b = 0; b < B; ++b) {
      c[static_cast<size_t>(b)] =
          initial_causal_colmajor_scalar(c, B, N, b, z, horizon);
    }
  }

  for (int n = 1; n < N; ++n) {
    double* cur = c.data() + static_cast<size_t>(n) * Bs;
    const double* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] += z * prev[static_cast<size_t>(b)];
    }
  }

  double* last = c.data() + static_cast<size_t>(N - 1) * Bs;
  const double* before_last = c.data() + static_cast<size_t>(N - 2) * Bs;
  const double denom = z * z - 1.0;
  for (int b = 0; b < B; ++b) {
    last[static_cast<size_t>(b)] =
        (z * before_last[static_cast<size_t>(b)] +
         last[static_cast<size_t>(b)]) * z / denom;
  }

  for (int n = N - 2; n >= 0; --n) {
    double* cur = c.data() + static_cast<size_t>(n) * Bs;
    const double* next = c.data() + static_cast<size_t>(n + 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] =
          z * (next[static_cast<size_t>(b)] - cur[static_cast<size_t>(b)]);
    }
  }
}

static inline int mirror_symmetric_index(int k, int N)
{
  if (N <= 1) return 0;
  const int period = 2 * N - 2;
  int t = k % period;
  if (t < 0) t += period;
  return (t >= N) ? (period - t) : t;
}

template <typename Scalar>
static void symmetric_fir_half2_colmajor(
  const std::vector<Scalar>& c,
  int B,
  int N,
  double h0d,
  double h1d,
  std::vector<Scalar>& s)
{
  const size_t total =
      static_cast<size_t>(std::max(B, 0)) *
      static_cast<size_t>(std::max(N, 0));
  s.resize(total);
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  const Scalar h0 = static_cast<Scalar>(h0d);
  const Scalar h1 = static_cast<Scalar>(h1d);

  if (N == 1) {
    const Scalar* src = c.data();
    Scalar* dst = s.data();
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      dst[bi] = h0 * src[bi] + h1 * (src[bi] + src[bi]);
    }
    return;
  }

  {
    const Scalar* c0 = c.data();
    const Scalar* c1 = c.data() + Bs;
    Scalar* dst = s.data();
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      dst[bi] = h0 * c0[bi] + h1 * (c1[bi] + c1[bi]);
    }
  }

  for (int n = 1; n + 1 < N; ++n) {
    const Scalar* left = c.data() + static_cast<size_t>(n - 1) * Bs;
    const Scalar* center = c.data() + static_cast<size_t>(n) * Bs;
    const Scalar* right = c.data() + static_cast<size_t>(n + 1) * Bs;
    Scalar* dst = s.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      dst[bi] = h0 * center[bi] + h1 * (left[bi] + right[bi]);
    }
  }

  {
    const Scalar* cn = c.data() + static_cast<size_t>(N - 1) * Bs;
    const Scalar* cp = c.data() + static_cast<size_t>(N - 2) * Bs;
    Scalar* dst = s.data() + static_cast<size_t>(N - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      dst[bi] = h0 * cn[bi] + h1 * (cp[bi] + cp[bi]);
    }
  }
}

template <typename Scalar>
static void diff_sa_as_colmajor_pair(
  std::vector<Scalar>& c,
  int B,
  int N,
  std::vector<Scalar>& work)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  if (N == 1) {
    Scalar* first = c.data();
    for (int b = 0; b < B; ++b) {
      first[static_cast<size_t>(b)] *= static_cast<Scalar>(2);
    }
    return;
  }

  work.resize(Bs);
  const Scalar* first = c.data();
  for (int b = 0; b < B; ++b) {
    work[static_cast<size_t>(b)] = first[static_cast<size_t>(b)];
  }

  {
    Scalar* row0 = c.data();
    const Scalar* row1 = c.data() + Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const Scalar d0 = work[bi] - row1[bi];
      row0[bi] = d0 * static_cast<Scalar>(2);
    }
  }

  for (int n = 1; n + 1 < N; ++n) {
    Scalar* cur = c.data() + static_cast<size_t>(n) * Bs;
    const Scalar* next = c.data() + static_cast<size_t>(n + 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const Scalar orig_cur = cur[bi];
      cur[bi] = (orig_cur - next[bi]) - (work[bi] - orig_cur);
      work[bi] = orig_cur;
    }
  }

  {
    Scalar* last = c.data() + static_cast<size_t>(N - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const Scalar orig_last = last[bi];
      last[bi] = (orig_last - work[bi]) - (work[bi] - orig_last);
    }
  }
}

template <typename Scalar>
static void add_average_colmajor(
  std::vector<Scalar>& c,
  int B,
  int N,
  const std::vector<Scalar>& average)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  for (int n = 0; n < N; ++n) {
    Scalar* row = c.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      row[bi] += average[bi];
    }
  }
}

template <typename Scalar>
static void diff_as_colmajor_plain(
  std::vector<Scalar>& c,
  int B,
  int N)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  if (N == 1) {
    Scalar* first = c.data();
    for (int b = 0; b < B; ++b) {
      first[static_cast<size_t>(b)] *= static_cast<Scalar>(2);
    }
    return;
  }

  for (int n = N - 1; n > 0; --n) {
    Scalar* cur = c.data() + static_cast<size_t>(n) * Bs;
    const Scalar* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] -= prev[static_cast<size_t>(b)];
    }
  }

  Scalar* first = c.data();
  for (int b = 0; b < B; ++b) {
    first[static_cast<size_t>(b)] *= static_cast<Scalar>(2);
  }
}

template <typename Scalar>
static void diff_as_colmajor_add_average(
  std::vector<Scalar>& c,
  int B,
  int N,
  const std::vector<Scalar>& average)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  if (N == 1) {
    Scalar* first = c.data();
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      first[bi] = first[bi] * static_cast<Scalar>(2) + average[bi];
    }
    return;
  }

  for (int n = N - 1; n > 0; --n) {
    Scalar* cur = c.data() + static_cast<size_t>(n) * Bs;
    const Scalar* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      cur[bi] = cur[bi] - prev[bi] + average[bi];
    }
  }

  Scalar* first = c.data();
  for (int b = 0; b < B; ++b) {
    const size_t bi = static_cast<size_t>(b);
    first[bi] = first[bi] * static_cast<Scalar>(2) + average[bi];
  }
}

template <typename Scalar>
static void diff_sa_as_colmajor_pair_add_average(
  std::vector<Scalar>& c,
  int B,
  int N,
  const std::vector<Scalar>& average,
  std::vector<Scalar>& work)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  if (N == 1) {
    Scalar* first = c.data();
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      first[bi] = first[bi] * static_cast<Scalar>(2) + average[bi];
    }
    return;
  }

  work.resize(Bs);
  const Scalar* first = c.data();
  for (int b = 0; b < B; ++b) {
    work[static_cast<size_t>(b)] = first[static_cast<size_t>(b)];
  }

  {
    Scalar* row0 = c.data();
    const Scalar* row1 = c.data() + Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const Scalar d0 = work[bi] - row1[bi];
      row0[bi] = d0 * static_cast<Scalar>(2) + average[bi];
    }
  }

  for (int n = 1; n + 1 < N; ++n) {
    Scalar* cur = c.data() + static_cast<size_t>(n) * Bs;
    const Scalar* next = c.data() + static_cast<size_t>(n + 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const Scalar orig_cur = cur[bi];
      cur[bi] = (orig_cur - next[bi]) - (work[bi] - orig_cur) + average[bi];
      work[bi] = orig_cur;
    }
  }

  {
    Scalar* last = c.data() + static_cast<size_t>(N - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const Scalar orig_last = last[bi];
      last[bi] = (orig_last - work[bi]) - (work[bi] - orig_last) + average[bi];
    }
  }
}

template <typename Scalar>
static void do_diff_colmajor_add_average_impl(
  std::vector<Scalar>& c,
  int B,
  int N,
  int nb,
  const std::vector<Scalar>& average,
  std::vector<Scalar>& work)
{
  if (B <= 0 || N <= 0) return;
  if (nb <= 0) { add_average_colmajor(c, B, N, average); return; }
  if (nb == 1) { diff_as_colmajor_add_average(c, B, N, average); return; }
  if (nb == 2) {
    diff_sa_as_colmajor_pair_add_average(c, B, N, average, work);
    return;
  }
  if (nb == 3) {
    diff_as_colmajor_plain(c, B, N);
    diff_sa_as_colmajor_pair_add_average(c, B, N, average, work);
    return;
  }
  diff_sa_as_colmajor_pair(c, B, N, work);
  diff_sa_as_colmajor_pair_add_average(c, B, N, average, work);
}

static void average_colmajor(
  const std::vector<double>& c,
  int B,
  int N,
  std::vector<double>& average)
{
  average.assign(static_cast<size_t>(B), 0.0);
  if (B <= 0 || N <= 1) return;

  const size_t Bs = static_cast<size_t>(B);
  for (int n = 0; n < N; ++n) {
    const double* col = c.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      average[static_cast<size_t>(b)] += col[static_cast<size_t>(b)];
    }
  }

  const double denom = 2.0 * static_cast<double>(N) - 2.0;
  const double* first = c.data();
  const double* last = c.data() + static_cast<size_t>(N - 1) * Bs;
  for (int b = 0; b < B; ++b) {
    const size_t bi = static_cast<size_t>(b);
    average[bi] = (2.0 * average[bi] - last[bi] - first[bi]) / denom;
  }
}

static void integ_sa_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  const std::vector<double>& average)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  double* first = c.data();
  for (int b = 0; b < B; ++b) {
    const size_t bi = static_cast<size_t>(b);
    first[bi] = (first[bi] - average[bi]) * 0.5;
  }

  for (int n = 1; n < N; ++n) {
    double* cur = c.data() + static_cast<size_t>(n) * Bs;
    const double* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      cur[bi] = cur[bi] - average[bi] + prev[bi];
    }
  }
}

static void integ_as_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  std::vector<double>& work)
{
  if (B <= 0 || N <= 0) return;

  work.assign(static_cast<size_t>(B), 0.0);
  const size_t Bs = static_cast<size_t>(B);
  for (int n = 1; n < N; ++n) {
    double* cur = c.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const double tmp = cur[bi];
      cur[bi] = (n == 1) ? 0.0 : -work[bi];
      work[bi] += tmp;
    }
  }
}

static void diff_sa_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  std::vector<double>& work)
{
  if (B <= 0 || N < 2) return;

  work.assign(static_cast<size_t>(B), 0.0);
  const size_t Bs = static_cast<size_t>(B);
  const double* before_last = c.data() + static_cast<size_t>(N - 2) * Bs;
  for (int b = 0; b < B; ++b) {
    work[static_cast<size_t>(b)] = before_last[static_cast<size_t>(b)];
  }

  for (int n = 0; n + 1 < N; ++n) {
    double* cur = c.data() + static_cast<size_t>(n) * Bs;
    const double* next = c.data() + static_cast<size_t>(n + 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] -= next[static_cast<size_t>(b)];
    }
  }

  double* last = c.data() + static_cast<size_t>(N - 1) * Bs;
  for (int b = 0; b < B; ++b) {
    last[static_cast<size_t>(b)] -= work[static_cast<size_t>(b)];
  }
}

static void diff_as_colmajor(
  std::vector<double>& c,
  int B,
  int N)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  if (N == 1) {
    double* first = c.data();
    for (int b = 0; b < B; ++b) {
      first[static_cast<size_t>(b)] *= 2.0;
    }
    return;
  }

  for (int n = N - 1; n > 0; --n) {
    double* cur = c.data() + static_cast<size_t>(n) * Bs;
    const double* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] -= prev[static_cast<size_t>(b)];
    }
  }

  double* first = c.data();
  for (int b = 0; b < B; ++b) {
    first[static_cast<size_t>(b)] *= 2.0;
  }
}

void get_interpolation_coefficients_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  int deg)
{
  if (deg <= 1 || N <= 1 || B <= 0) return;

  const auto& poles = spline_poles(deg);
  double lambda = 1.0;
  for (double z : poles) {
    lambda *= (1.0 - z) * (1.0 - 1.0 / z);
  }
  for (double& v : c) {
    v *= lambda;
  }

  for (double z : poles) {
    apply_interpolation_pole_colmajor(
        c, B, N, z, initial_causal_horizon_colmajor(N, z));
  }
}

void apply_interpolation_poles_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  int deg)
{
  if (deg <= 1 || N <= 1 || B <= 0) return;

  const auto& poles = spline_poles(deg);
  for (double z : poles) {
    apply_interpolation_pole_colmajor(
        c, B, N, z, initial_causal_horizon_colmajor(N, z));
  }
}

void do_integ_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  int nb,
  std::vector<double>& average,
  std::vector<double>& work)
{
  if (B <= 0) {
    average.clear();
    return;
  }
  if (N <= 1 || nb <= 0) {
    average.assign(static_cast<size_t>(B), 0.0);
    return;
  }

  if (nb >= 1) {
    average_colmajor(c, B, N, average);
    integ_sa_colmajor(c, B, N, average);
  }
  if (nb >= 2) {
    integ_as_colmajor(c, B, N, work);
  }
  if (nb >= 3) {
    average_colmajor(c, B, N, work);
    integ_sa_colmajor(c, B, N, work);
  }
  if (nb >= 4) {
    integ_as_colmajor(c, B, N, work);
  }
}

void do_diff_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  int nb,
  std::vector<double>& work)
{
  if (B <= 0 || N <= 0 || nb <= 0) return;
  if (nb == 1) { diff_as_colmajor(c, B, N); return; }
  if (nb == 2) { diff_sa_as_colmajor_pair(c, B, N, work); return; }
  if (nb == 3) { diff_as_colmajor(c, B, N); diff_sa_as_colmajor_pair(c, B, N, work); return; }
  diff_sa_as_colmajor_pair(c, B, N, work);
  diff_sa_as_colmajor_pair(c, B, N, work);
}

void do_diff_colmajor_add_average(
  std::vector<double>& c,
  int B,
  int N,
  int nb,
  const std::vector<double>& average,
  std::vector<double>& work)
{
  do_diff_colmajor_add_average_impl(c, B, N, nb, average, work);
}

void get_samples_colmajor(
  std::vector<double>& c,
  int B,
  int N,
  int deg,
  std::vector<double>& work)
{
  if (deg <= 1 || B <= 0 || N <= 0) return;

  const auto& h = sampling_fir(deg);
  if (h.empty()) return;

  if (h.size() == 2) {
    symmetric_fir_half2_colmajor(c, B, N, h[0], h[1], work);
    c.swap(work);
    return;
  }

  const size_t total = static_cast<size_t>(B) * static_cast<size_t>(N);
  work.resize(total);

  const size_t Bs = static_cast<size_t>(B);
  for (int n = 0; n < N; ++n) {
    const double* center = c.data() + static_cast<size_t>(n) * Bs;
    double* dst = work.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = h[0] * center[static_cast<size_t>(b)];
    }

    for (int j = 1; j < static_cast<int>(h.size()); ++j) {
      const int left_idx = mirror_symmetric_index(n - j, N);
      const int right_idx = mirror_symmetric_index(n + j, N);
      const double* left = c.data() + static_cast<size_t>(left_idx) * Bs;
      const double* right = c.data() + static_cast<size_t>(right_idx) * Bs;
      const double hj = h[static_cast<size_t>(j)];
      for (int b = 0; b < B; ++b) {
        const size_t bi = static_cast<size_t>(b);
        dst[bi] += hj * (left[bi] + right[bi]);
      }
    }
  }

  c.swap(work);
}

static float initial_causal_colmajor_scalar_f32(
  const std::vector<float>& c,
  int B,
  int N,
  int b,
  float z,
  size_t horizon)
{
  if (N == 0) return 0.0f;
  if (N == 1) return c[static_cast<size_t>(b)];

  const size_t Bs = static_cast<size_t>(B);
  const size_t bi = static_cast<size_t>(b);

  if (horizon < static_cast<size_t>(N)) {
    const float* ptr = c.data() + bi;
    float sum = *ptr;
    float p = z;
    for (size_t n = 1; n < horizon; ++n) {
      ptr += Bs;
      sum += p * (*ptr);
      p *= z;
    }
    return sum;
  }

  const float zn = std::pow(z, static_cast<float>(N - 1));
  const float* ptr = c.data() + bi;
  float sum = *ptr + zn * (*(ptr + static_cast<size_t>(N - 1) * Bs));
  float p1 = z;
  float p2 = (zn * zn) / z;
  for (int n = 1; n + 1 < N; ++n) {
    ptr += Bs;
    sum += (p1 + p2) * (*ptr);
    p1 *= z;
    p2 /= z;
  }

  return sum / (1.0f - (zn * zn));
}

static inline size_t initial_causal_horizon_colmajor_f32(
  int N,
  float z,
  float tol = 1e-10f)
{
  size_t horizon = static_cast<size_t>(std::max(N, 0));
  if (N > 0 && tol > 0.0f) {
    horizon = std::min(
        static_cast<size_t>(N),
        static_cast<size_t>(2 + std::log(tol) / std::log(std::abs(z))));
  }
  return horizon;
}

static void apply_interpolation_pole_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  float z,
  size_t horizon)
{
  const size_t Bs = static_cast<size_t>(B);
  if (rowwise_initial_causal_enabled()) {
    float* first = c.data();
    if (horizon < static_cast<size_t>(N)) {
      float p = z;
      for (size_t n = 1; n < horizon; ++n) {
        const float* row = c.data() + n * Bs;
        for (int b = 0; b < B; ++b) {
          first[static_cast<size_t>(b)] += p * row[static_cast<size_t>(b)];
        }
        p *= z;
      }
    } else {
      const float zn = std::pow(z, static_cast<float>(N - 1));
      const float* last = c.data() + static_cast<size_t>(N - 1) * Bs;
      for (int b = 0; b < B; ++b) {
        first[static_cast<size_t>(b)] += zn * last[static_cast<size_t>(b)];
      }
      float p1 = z;
      float p2 = (zn * zn) / z;
      for (int n = 1; n + 1 < N; ++n) {
        const float* row = c.data() + static_cast<size_t>(n) * Bs;
        const float w = p1 + p2;
        for (int b = 0; b < B; ++b) {
          first[static_cast<size_t>(b)] += w * row[static_cast<size_t>(b)];
        }
        p1 *= z;
        p2 /= z;
      }
      const float denom = 1.0f - (zn * zn);
      for (int b = 0; b < B; ++b) {
        first[static_cast<size_t>(b)] /= denom;
      }
    }
  } else {
    for (int b = 0; b < B; ++b) {
      c[static_cast<size_t>(b)] =
          initial_causal_colmajor_scalar_f32(c, B, N, b, z, horizon);
    }
  }

  for (int n = 1; n < N; ++n) {
    float* cur = c.data() + static_cast<size_t>(n) * Bs;
    const float* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] += z * prev[static_cast<size_t>(b)];
    }
  }

  float* last = c.data() + static_cast<size_t>(N - 1) * Bs;
  const float* before_last = c.data() + static_cast<size_t>(N - 2) * Bs;
  const float denom = z * z - 1.0f;
  for (int b = 0; b < B; ++b) {
    last[static_cast<size_t>(b)] =
        (z * before_last[static_cast<size_t>(b)] +
         last[static_cast<size_t>(b)]) * z / denom;
  }

  for (int n = N - 2; n >= 0; --n) {
    float* cur = c.data() + static_cast<size_t>(n) * Bs;
    const float* next = c.data() + static_cast<size_t>(n + 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] =
          z * (next[static_cast<size_t>(b)] - cur[static_cast<size_t>(b)]);
    }
  }
}

static void average_colmajor_f32(
  const std::vector<float>& c,
  int B,
  int N,
  std::vector<float>& average)
{
  average.assign(static_cast<size_t>(B), 0.0f);
  if (B <= 0 || N <= 1) return;

  const size_t Bs = static_cast<size_t>(B);
  for (int n = 0; n < N; ++n) {
    const float* col = c.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      average[static_cast<size_t>(b)] += col[static_cast<size_t>(b)];
    }
  }

  const float denom = 2.0f * static_cast<float>(N) - 2.0f;
  const float* first = c.data();
  const float* last = c.data() + static_cast<size_t>(N - 1) * Bs;
  for (int b = 0; b < B; ++b) {
    const size_t bi = static_cast<size_t>(b);
    average[bi] = (2.0f * average[bi] - last[bi] - first[bi]) / denom;
  }
}

static void integ_sa_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  const std::vector<float>& average)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  float* first = c.data();
  for (int b = 0; b < B; ++b) {
    const size_t bi = static_cast<size_t>(b);
    first[bi] = (first[bi] - average[bi]) * 0.5f;
  }

  for (int n = 1; n < N; ++n) {
    float* cur = c.data() + static_cast<size_t>(n) * Bs;
    const float* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      cur[bi] = cur[bi] - average[bi] + prev[bi];
    }
  }
}

static void integ_as_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  std::vector<float>& work)
{
  if (B <= 0 || N <= 0) return;

  work.assign(static_cast<size_t>(B), 0.0f);
  const size_t Bs = static_cast<size_t>(B);
  for (int n = 1; n < N; ++n) {
    float* cur = c.data() + static_cast<size_t>(n) * Bs;
    for (int b = 0; b < B; ++b) {
      const size_t bi = static_cast<size_t>(b);
      const float tmp = cur[bi];
      cur[bi] = (n == 1) ? 0.0f : -work[bi];
      work[bi] += tmp;
    }
  }
}

static void diff_sa_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  std::vector<float>& work)
{
  if (B <= 0 || N < 2) return;

  work.assign(static_cast<size_t>(B), 0.0f);
  const size_t Bs = static_cast<size_t>(B);
  const float* before_last = c.data() + static_cast<size_t>(N - 2) * Bs;
  for (int b = 0; b < B; ++b) {
    work[static_cast<size_t>(b)] = before_last[static_cast<size_t>(b)];
  }

  for (int n = 0; n + 1 < N; ++n) {
    float* cur = c.data() + static_cast<size_t>(n) * Bs;
    const float* next = c.data() + static_cast<size_t>(n + 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] -= next[static_cast<size_t>(b)];
    }
  }

  float* last = c.data() + static_cast<size_t>(N - 1) * Bs;
  for (int b = 0; b < B; ++b) {
    last[static_cast<size_t>(b)] -= work[static_cast<size_t>(b)];
  }
}

static void diff_as_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N)
{
  if (B <= 0 || N <= 0) return;

  const size_t Bs = static_cast<size_t>(B);
  if (N == 1) {
    float* first = c.data();
    for (int b = 0; b < B; ++b) {
      first[static_cast<size_t>(b)] *= 2.0f;
    }
    return;
  }

  for (int n = N - 1; n > 0; --n) {
    float* cur = c.data() + static_cast<size_t>(n) * Bs;
    const float* prev = c.data() + static_cast<size_t>(n - 1) * Bs;
    for (int b = 0; b < B; ++b) {
      cur[static_cast<size_t>(b)] -= prev[static_cast<size_t>(b)];
    }
  }

  float* first = c.data();
  for (int b = 0; b < B; ++b) {
    first[static_cast<size_t>(b)] *= 2.0f;
  }
}

void get_interpolation_coefficients_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  int deg)
{
  if (deg <= 1 || N <= 1 || B <= 0) return;

  const auto& poles = spline_poles(deg);
  float lambda = 1.0f;
  for (double zd : poles) {
    const float z = static_cast<float>(zd);
    lambda *= (1.0f - z) * (1.0f - 1.0f / z);
  }
  for (float& v : c) {
    v *= lambda;
  }

  for (double zd : poles) {
    const float z = static_cast<float>(zd);
    apply_interpolation_pole_colmajor_f32(
        c, B, N, z, initial_causal_horizon_colmajor_f32(N, z));
  }
}

void apply_interpolation_poles_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  int deg)
{
  if (deg <= 1 || N <= 1 || B <= 0) return;

  const auto& poles = spline_poles(deg);
  for (double zd : poles) {
    const float z = static_cast<float>(zd);
    apply_interpolation_pole_colmajor_f32(
        c, B, N, z, initial_causal_horizon_colmajor_f32(N, z));
  }
}

void do_integ_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  int nb,
  std::vector<float>& average,
  std::vector<float>& work)
{
  if (B <= 0) {
    average.clear();
    return;
  }
  if (N <= 1 || nb <= 0) {
    average.assign(static_cast<size_t>(B), 0.0f);
    return;
  }

  if (nb >= 1) {
    average_colmajor_f32(c, B, N, average);
    integ_sa_colmajor_f32(c, B, N, average);
  }
  if (nb >= 2) {
    integ_as_colmajor_f32(c, B, N, work);
  }
  if (nb >= 3) {
    average_colmajor_f32(c, B, N, work);
    integ_sa_colmajor_f32(c, B, N, work);
  }
  if (nb >= 4) {
    integ_as_colmajor_f32(c, B, N, work);
  }
}

void do_diff_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  int nb,
  std::vector<float>& work)
{
  if (B <= 0 || N <= 0 || nb <= 0) return;
  if (nb == 1) { diff_as_colmajor_f32(c, B, N); return; }
  if (nb == 2) { diff_sa_as_colmajor_pair(c, B, N, work); return; }
  if (nb == 3) { diff_as_colmajor_f32(c, B, N); diff_sa_as_colmajor_pair(c, B, N, work); return; }
  diff_sa_as_colmajor_pair(c, B, N, work);
  diff_sa_as_colmajor_pair(c, B, N, work);
}

void do_diff_colmajor_add_average_f32(
  std::vector<float>& c,
  int B,
  int N,
  int nb,
  const std::vector<float>& average,
  std::vector<float>& work)
{
  do_diff_colmajor_add_average_impl(c, B, N, nb, average, work);
}

void get_samples_colmajor_f32(
  std::vector<float>& c,
  int B,
  int N,
  int deg,
  std::vector<float>& work)
{
  if (deg <= 1 || B <= 0 || N <= 0) return;

  const auto& h = sampling_fir(deg);
  if (h.empty()) return;

  if (h.size() == 2) {
    symmetric_fir_half2_colmajor(c, B, N, h[0], h[1], work);
    c.swap(work);
    return;
  }

  const size_t total = static_cast<size_t>(B) * static_cast<size_t>(N);
  work.resize(total);

  const size_t Bs = static_cast<size_t>(B);
  for (int n = 0; n < N; ++n) {
    const float* center = c.data() + static_cast<size_t>(n) * Bs;
    float* dst = work.data() + static_cast<size_t>(n) * Bs;
    const float h0 = static_cast<float>(h[0]);
    for (int b = 0; b < B; ++b) {
      dst[static_cast<size_t>(b)] = h0 * center[static_cast<size_t>(b)];
    }

    for (int j = 1; j < static_cast<int>(h.size()); ++j) {
      const int left_idx = mirror_symmetric_index(n - j, N);
      const int right_idx = mirror_symmetric_index(n + j, N);
      const float* left = c.data() + static_cast<size_t>(left_idx) * Bs;
      const float* right = c.data() + static_cast<size_t>(right_idx) * Bs;
      const float hj = static_cast<float>(h[static_cast<size_t>(j)]);
      for (int b = 0; b < B; ++b) {
        const size_t bi = static_cast<size_t>(b);
        dst[bi] += hj * (left[bi] + right[bi]);
      }
    }
  }

  c.swap(work);
}

} // namespace lsresize
