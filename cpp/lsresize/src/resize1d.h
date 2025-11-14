// splineops/cpp/lsresize/src/resize1d.h
#pragma once
#include <vector>
#include <cstdint>

namespace lsresize {

struct LSParams {
  int    interp_degree;   // n
  int    analy_degree;    // n1  (=-1 for pure interpolation)
  int    synthe_degree;   // n2  (usually = n)
  double zoom;            // a
  double shift;           // b
  bool   inversable;      // size adjustment
};

// Precomputed, per-axis resampling plan.
// Reused for every 1-D line with the same (N, zoom, degrees, inversable, shift).
struct Plan1D {
  int  N;               // input line length
  int  outN;            // true output length
  int  out_total;       // output length incl. tail (add_border)
  int  length_total;    // extended input length (N + ceil(add_border/zoom))
  bool symmetric_ext;   // boundary for negative indices (true=symmetric, false=antisymmetric)

  // CSR-style layout for variable window sizes per output position l
  // row_ptr.size() == out_total + 1; for each row l, weights[row_ptr[l] ... row_ptr[l+1]-1]
  std::vector<int>    row_ptr;     // offsets into weights (contiguous per row)
  std::vector<double> weights;     // fact * beta(x - k, total_degree)

  // Per-row window metadata for contiguous access
  std::vector<int> kmin;           // window start index kmin[l]
  std::vector<int> win_len;        // window length M[l] = kmax - kmin + 1

  // Global pads to build a single contiguous extended buffer
  int left_pad  = 0;               // max(0, -min_kmin across rows)
  int right_pad = 0;               // max(0,  max_kmax - (length_total-1))

  // Precomputed left-pad mapping for negative indices: -t -> sign * coeff[src]
  // (size == left_pad). This removes per-line mirror math.
  std::vector<int>  pad_src_idx;   // source index in coeff (clamped later to [0, N-1])
  std::vector<char> pad_src_sgn;   // +1 / -1

  // Precomputed right-tail mapping: ext[N + i] = rp_sign * coeff[rp_src[i]]
  std::vector<int>  rp_src;        // size == max(0, length_total - N)
  char              rp_sign = 1;
};

// Per-thread reusable workspace to avoid per-line allocations
struct Work1D {
  std::vector<double> coeff;     // interpolation coefficients
  std::vector<double> ext;       // finite extension (right tail only)
  std::vector<double> ext_full;  // [left_pad | ext | right_pad] single buffer
  std::vector<double> y;         // accumulator / tail buffer
};

// Build the reusable plan once per axis.
Plan1D make_plan_1d(int N, const LSParams& p);

// Allocation-free fast path: reuse the provided workspace.
void resize_1d_ws(const std::vector<double>& in,
                  std::vector<double>& out,
                  const LSParams& p,
                  const Plan1D& plan,
                  Work1D& ws);

} // namespace lsresize
