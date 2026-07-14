// Standalone plan-build microbenchmark for experiment_resize_stable_ls.py.
// This file is not linked into splineops.  Build with:
//   c++ -O3 -DNDEBUG -std=c++17 scripts/experiment_resize_stable_ls_plan.cpp -o /tmp/stable_ls_plan

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double beta(double x, int n) {
  x = std::abs(x);
  if (n == 0) return x < 0.5 ? 1.0 : 0.0;
  if (n == 1) return x < 1.0 ? 1.0 - x : 0.0;
  if (n == 2) {
    if (x < 0.5) return 0.75 - x*x;
    if (x < 1.5) { x -= 1.5; return 0.5*x*x; }
    return 0.0;
  }
  if (x < 1.0) return 0.5*x*x*(x - 2.0) + 2.0/3.0;
  if (x < 2.0) { x -= 2.0; return -x*x*x/6.0; }
  return 0.0;
}

struct Rule { int count; std::array<double, 4> x, w; };

static Rule rule(int order) {
  if (order == 1) return {1, {0,0,0,0}, {2,0,0,0}};
  if (order == 2) {
    const double x = 1/std::sqrt(3.0);
    return {2, {-x,x,0,0}, {1,1,0,0}};
  }
  if (order == 3) {
    const double x = std::sqrt(3.0/5.0);
    return {3, {-x,0,x,0}, {5.0/9,8.0/9,5.0/9,0}};
  }
  const double s = std::sqrt(6.0/5.0);
  const double x0 = std::sqrt((3.0 + 2.0*s)/7.0);
  const double x1 = std::sqrt((3.0 - 2.0*s)/7.0);
  const double w0 = (18.0 - std::sqrt(30.0))/36.0;
  const double w1 = (18.0 + std::sqrt(30.0))/36.0;
  return {4, {-x0,-x1,x1,x0}, {w0,w1,w1,w0}};
}

static double cross_gauss(double x, double a, int n, int m) {
  const double rn = 0.5*(n+1), rm = 0.5*(m+1);
  const double lo = std::max(-a*rn, x-rm);
  const double hi = std::min( a*rn, x+rm);
  if (!(lo < hi)) return 0.0;
  std::array<double, 12> knots{};
  int count = 0;
  knots[count++] = lo; knots[count++] = hi;
  for (int i=0; i<=n+1; ++i) {
    const double v = a*(-rn+i);
    if (v > lo && v < hi) knots[count++] = v;
  }
  for (int j=0; j<=m+1; ++j) {
    const double v = x-(-rm+j);
    if (v > lo && v < hi) knots[count++] = v;
  }
  std::sort(knots.begin(), knots.begin()+count);
  int unique = 1;
  for (int i=1; i<count; ++i)
    if (knots[i] != knots[unique-1]) knots[unique++] = knots[i];
  const Rule q = rule((n+m+2)/2);
  double total = 0.0;
  for (int i=0; i+1<unique; ++i) {
    const double mid = 0.5*(knots[i]+knots[i+1]);
    const double half = 0.5*(knots[i+1]-knots[i]);
    for (int j=0; j<q.count; ++j) {
      const double t = mid + half*q.x[j];
      total += half*q.w[j]*beta(t/a,n)*beta(x-t,m);
    }
  }
  return total;
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 65536;
  const double zoom = argc > 2 ? std::atof(argv[2]) : 0.37;
  const int n = argc > 3 ? std::atoi(argv[3]) : 3;
  const int m = argc > 4 ? std::atoi(argv[4]) : n;
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;
  const int M = std::max(1, static_cast<int>(std::llround(N*zoom)));
  const double a = M > 1 ? double(M-1)/double(N-1) : 1.0;
  const double radius = 0.5*((n+1)*a + m+1);
  double best = 1e300, checksum = 0.0;
  std::size_t nnz = 0;
  for (int repeat=0; repeat<repeats; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<double> weights;
    weights.reserve(static_cast<std::size_t>((n+1)*M + (m+1)*N + 16));
    for (int l=0; l<M; ++l) {
      const int first = static_cast<int>(std::ceil((l-radius)/a));
      const int last  = static_cast<int>(std::floor((l+radius)/a));
      for (int k=first; k<=last; ++k) {
        const double value = cross_gauss(l-a*k, a, n, m);
        if (value > 0.0) weights.push_back(value);
      }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double,std::milli>(stop-start).count();
    best = std::min(best, ms);
    nnz = weights.size();
    checksum = 0.0;
    for (double value : weights) checksum += value;
  }
  std::printf("N=%d M=%d a=%.9g n=%d m=%d nnz=%zu build_ms=%.6f checksum=%.17g\n",
              N,M,a,n,m,nnz,best,checksum);
}
