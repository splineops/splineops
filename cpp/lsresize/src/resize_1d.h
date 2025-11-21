// splineops/cpp/lsresize/src/resize_1d.h
#pragma once
#include <vector>
#include <cstdint>

namespace lsresize {

struct LSParams {
  int    interp_degree;
  int    analy_degree;
  int    synthe_degree;
  double zoom;
  double shift;
  bool   inversable;
};

// Templated Plan1D over scalar Real
template <typename Real>
struct Plan1D_T {
  int  N;
  int  outN;
  int  out_total;
  int  length_total;
  bool symmetric_ext;

  std::vector<int>    row_ptr;
  std::vector<Real>   weights;

  std::vector<int> kmin;
  std::vector<int> win_len;

  int left_pad  = 0;
  int right_pad = 0;

  std::vector<int>  pad_src_idx;
  std::vector<char> pad_src_sgn;

  std::vector<int>  rp_src;
  char              rp_sign = 1;
};

// Templated workspace
template <typename Real>
struct Work1D_T {
  std::vector<Real> coeff;
  std::vector<Real> ext;
  std::vector<Real> ext_full;
  std::vector<Real> y;
};

// Default aliases: keep existing names as double-based
using Plan1D  = Plan1D_T<double>;
using Work1D  = Work1D_T<double>;

// Build the reusable plan once per axis.
Plan1D make_plan_1d(int N, const LSParams& p);

// Allocation-free fast paths (double-only API as before)
void resize_1d_ws(const std::vector<double>& in,
                  std::vector<double>& out,
                  const LSParams& p,
                  const Plan1D& plan,
                  Work1D& ws);

void resize_1d_ws_raw(const double* in,
                      double* out,
                      const LSParams& p,
                      const Plan1D& plan,
                      Work1D& ws);

} // namespace lsresize
