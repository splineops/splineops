// splineops/cpp/lsresize/src/resize1d.h
#pragma once
#include <vector>
#include <cstdint>

namespace lsresize {

struct LSParams {
  int interp_degree;   // n
  int analy_degree;    // n1  (=-1 for pure interpolation)
  int synthe_degree;   // n2  (usually = n)
  double zoom;         // a
  double shift;        // b
  bool inversable;     // size adjustment
};

// Precomputed, per-axis resampling plan.
// Reused for every 1-D line with the same (N, zoom, degrees, inversable, shift).
struct Plan1D {
  int N;                // input line length
  int outN;             // true output length
  int out_total;        // output length incl. tail (add_border)
  int length_total;     // extended input length (N + ceil(add_border/zoom))
  bool symmetric_ext;   // boundary model for negative indices (true=symmetric, false=antisymmetric)

  // CSR-style layout for variable window sizes per output position l
  // row_ptr.size() == out_total + 1
  std::vector<int> row_ptr;
  // For each nonzero (k in support of row l), we store:
  std::vector<int>        ext_index; // index into extended input [0 .. length_total-1]
  std::vector<signed char> ext_sign; // +1 or -1 (only non-1 for k<0 with antisymmetric)
  std::vector<double>     weights;   // fact * beta(x - k, total_degree)
};

// Legacy entry (kept for API compatibility)
void resize_1d(const std::vector<double>& in,
               std::vector<double>& out,
               const LSParams& p);

// Build a reusable plan once per axis
Plan1D make_plan_1d(int N, const LSParams& p);

// Fast path using a precomputed plan
void resize_1d_planned(const std::vector<double>& in,
                       std::vector<double>& out,
                       const LSParams& p,
                       const Plan1D& plan);

} // namespace lsresize
