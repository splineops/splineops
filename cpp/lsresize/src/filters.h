// splineops/cpp/lsresize/src/filters.h
#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace lsresize {

// poles z_k for degrees 2..7 (Unser '93)
inline const std::vector<double>& spline_poles(int deg) {
  static const std::vector<double> z2 = { std::sqrt(8.0) - 3.0 };
  static const std::vector<double> z3 = { std::sqrt(3.0) - 2.0 };
  static const std::vector<double> z4 = {
    std::sqrt(664.0 - std::sqrt(438976.0)) + std::sqrt(304.0) - 19.0,
    std::sqrt(664.0 + std::sqrt(438976.0)) - std::sqrt(304.0) - 19.0
  };
  static const std::vector<double> z5 = {
    std::sqrt(135.0/2.0 - std::sqrt(17745.0/4.0)) + std::sqrt(105.0/4.0) - 6.5,
    std::sqrt(135.0/2.0 + std::sqrt(17745.0/4.0)) - std::sqrt(105.0/4.0) - 6.5
  };
  static const std::vector<double> z6 = {
   -0.488294589303044755130118038883789062112279161239377608394,
   -0.081679271076237512597937765737059080653379610398148178525368,
   -0.00141415180832581775108724397655859252786416905534669851652709
  };
  static const std::vector<double> z7 = {
   -0.5352804307964381655424037816816460718339231523426924148812,
   -0.122554615192326690515272264359357343605486549427295558490763,
   -0.0091486948096082769285930216516478534156925639545994482648003
  };
  switch (deg) {
    case 0: case 1: { static const std::vector<double> none; return none; }
    case 2: return z2; case 3: return z3; case 4: return z4; case 5: return z5;
    case 6: return z6; case 7: return z7;
    default: throw std::invalid_argument("Invalid spline degree for poles [0..7]");
  }
}

// symmetric FIR taps for sampling (Step 5)
inline const std::vector<double>& sampling_fir(int deg) {
  static const std::vector<double> h2 = { 3.0/4.0, 1.0/8.0 };
  static const std::vector<double> h3 = { 2.0/3.0, 1.0/6.0 };
  static const std::vector<double> h4 = { 115.0/192.0, 19.0/96.0, 1.0/384.0 };
  static const std::vector<double> h5 = { 11.0/20.0, 13.0/60.0, 1.0/120.0 };
  static const std::vector<double> h6 = { 5887.0/11520.0, 10543.0/46080.0, 361.0/23040.0, 1.0/46080.0 };
  static const std::vector<double> h7 = { 151.0/315.0, 397.0/1680.0, 1.0/42.0, 1.0/5040.0 };
  switch (deg) {
    case 0: case 1: { static const std::vector<double> none; return none; }
    case 2: return h2; case 3: return h3; case 4: return h4; case 5: return h5;
    case 6: return h6; case 7: return h7;
    default: throw std::invalid_argument("Invalid degree for sampling FIR [0..7]");
  }
}

// spline IIR prefilter (Unser '93)
double initial_causal(const std::vector<double>& c, double z, double tol = 1e-10);
double initial_anti_causal(const std::vector<double>& c, double z);

inline void get_interpolation_coefficients(std::vector<double>& c, int deg) {
  if (deg <= 1 || c.size() <= 1) return;
  const auto& poles = spline_poles(deg);

  double lambda = 1.0; // normalization
  for (double z : poles) lambda *= (1.0 - z) * (1.0 - 1.0/z);
  for (double& v : c) v *= lambda;

  for (double z : poles) {
    // forward (causal)
    c[0] = initial_causal(c, z);
    for (size_t n = 1; n < c.size(); ++n)
      c[n] += z * c[n-1];

    // backward (anti-causal) — must include n == 0
    c.back() = initial_anti_causal(c, z);
    for (int n = static_cast<int>(c.size()) - 2; n >= 0; --n)
      c[n] = z * (c[n+1] - c[n]);
  }
}

// symmetric FIR sampling (Step 5)
void symmetric_fir(const std::vector<double>& h, const std::vector<double>& c, std::vector<double>& s);

inline void get_samples(std::vector<double>& c, int deg) {
  if (deg <= 1) return;
  const auto& h = sampling_fir(deg);
  std::vector<double> s(c.size(), 0.0);
  symmetric_fir(h, c, s);
  c.swap(s);
}

// running sums Δ^{-1} and centered differences Δ with alternation (Fig. 8)
double do_integ(std::vector<double>& c, int nb); // returns average to add back later
void do_diff(std::vector<double>& c, int nb);

// helpers
void integ_sa(std::vector<double>& c, double m);
void integ_as(const std::vector<double>& c, std::vector<double>& y);
void diff_sa(std::vector<double>& c);
void diff_as(std::vector<double>& c);

} // namespace lsresize
