// splineops/cpp/lsresize/src/resizend.h
#pragma once
#include <vector>
#include <cstdint>
#include "resize1d.h"

namespace lsresize {

// Process along a single axis by copying lines to/from temporary buffers
void resize_along_axis(const double* in, double* out,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& out_shape,
                       int axis,
                       const LSParams& p);

} // namespace lsresize
