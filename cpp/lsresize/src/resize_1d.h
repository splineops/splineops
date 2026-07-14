// splineops/cpp/lsresize/src/resize_1d.h
#pragma once
#include <memory>
#include <vector>
#include <cstdint>

// All 1-D entry points take a Work1D& workspace argument,
// which holds reusable scratch buffers to avoid per-line allocations.
// You must create one Work1D per thread and reuse it across calls.

namespace lsresize {

struct LSParams {
  int    interp_degree;   // n
  int    analy_degree;    // n1  (=-1 for pure interpolation)
  int    synthe_degree;   // n2  (usually = n)
  double zoom;            // a
  double shift;           // b
};

struct RowRun1D {
  int begin = 0;
  int end = 0;
  char interior = 0;
};

// Precomputed, per-axis resampling plan.
// Reused for every 1-D line with the same realized grid, degrees, and shift.
struct Plan1D {
  int  N;               // input line length
  int  outN;            // true output length
  int  out_total;       // output length incl. tail (add_border)
  int  length_total;    // materialized input/extension length (N when direct)
  double effective_zoom = 1.0; // (outN-1)/(N-1), non-degenerate axes only
  bool symmetric_ext;   // boundary for negative indices (true=symmetric, false=antisymmetric)
  bool direct_projection = false; // stable compact cross-Gram projection

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

  // Precomputed left-pad mapping for negative indices: -t -> sign * line[src]
  // (size == left_pad). This removes per-line mirror math.
  std::vector<int>  pad_src_idx;   // source index in line
  std::vector<char> pad_src_sgn;   // +1 / -1

  // Precomputed right-tail mapping: ext[N + i] = rp_sign * line[rp_src[i]]
  std::vector<int>  rp_src;        // size == max(0, length_total - N)
  char              rp_sign = 1;

  // Batched ND row map: direct interior output rows use coefficient columns
  // directly; boundary rows use coeff_src/coeff_sgn to preserve exact extension.
  std::vector<RowRun1D> row_runs;
  std::vector<int>      coeff_src;
  std::vector<std::int8_t> coeff_sgn;
  int                   interior_rows = 0;
  int                   mapped_rows = 0;

  // Compact exact-linear map for direct/fused interpolation kernels. These are
  // populated only for pure linear interpolation plans with support <= 3.
  bool direct_linear_ok = false;
  std::vector<unsigned char> direct_linear_count;
  std::vector<int>           direct_linear_src0;
  std::vector<int>           direct_linear_src1;
  std::vector<int>           direct_linear_src2;
  std::vector<double>        direct_linear_w0;
  std::vector<double>        direct_linear_w1;
  std::vector<double>        direct_linear_w2;
  std::vector<float>         direct_linear_w0_f32;
  std::vector<float>         direct_linear_w1_f32;
  std::vector<float>         direct_linear_w2_f32;
};

// Per-thread reusable workspace to avoid per-line allocations
struct Work1D {
  std::vector<double> line;      // input samples / spline coefficients
  std::vector<double> ext_full;  // [left_pad | ext | right_pad]
  std::vector<double> y;         // accumulator / tail buffer
};

// Build the reusable plan once per axis.
Plan1D make_plan_1d(int N, const LSParams& p);

// Process-local count- and byte-bounded cache for repeated axes. Set either
// LSRESIZE_PLAN_CACHE_SIZE=0 or LSRESIZE_PLAN_CACHE_BYTES=0 to disable it.
std::shared_ptr<const Plan1D> get_plan_1d_cached(int N, const LSParams& p);

// Allocation-free fast path: reuse the provided workspace (vector in/out).
void resize_1d_workspace(
  const std::vector<double>& in,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace);

// Allocation-free fast path for contiguous raw buffers.
void resize_1d_line_contiguous(
  const double* in,
  double* out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace);

// Allocation-free fast path when the caller has already filled `line`
// with N samples (plan.N). This avoids the extra copy from `in` into
// `workspace.line` and is used by the ND kernel fallback.
void resize_1d_line_buffered(
  std::vector<double>& line,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace);

} // namespace lsresize
