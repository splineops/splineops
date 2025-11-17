// splineops/cpp/lsresize/src/resizend.h
#pragma once
#include <vector>
#include <cstdint>
#include "resize1d.h"

namespace lsresize {

#ifndef LS_RESTRICT
#  if defined(_MSC_VER)
#    define LS_RESTRICT __restrict
#  else
#    define LS_RESTRICT __restrict__
#  endif
#endif

void resize_along_axis(const double* LS_RESTRICT in, double* LS_RESTRICT out,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& out_shape,
                       int axis,
                       const LSParams& p);

void resize_along_axis_f32(const float* LS_RESTRICT in, float* LS_RESTRICT out,
                           const std::vector<int64_t>& in_shape,
                           const std::vector<int64_t>& out_shape,
                           int axis,
                           const LSParams& p);

} // namespace lsresize
