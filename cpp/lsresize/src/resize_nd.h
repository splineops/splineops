// splineops/cpp/lsresize/src/resize_nd.h
#pragma once
#include <vector>
#include <cstdint>
#include "resize_1d.h"

namespace lsresize {

#ifndef LS_RESTRICT
#  if defined(_MSC_VER)
#    define LS_RESTRICT __restrict
#  else
#    define LS_RESTRICT __restrict__
#  endif
#endif

void resize_along_axis(
    const double* LS_RESTRICT in, 
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    int axis,
    const LSParams& p);

void resize_along_axis_f32(
    const float* LS_RESTRICT in, 
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    int axis,
    const LSParams& p);

// Execute an axis pass with an immutable plan supplied by the caller.  This is
// the reusable-plan entry point: unlike resize_along_axis(), it never consults
// the process-wide Plan1D cache.  The plan must have been built for the input
// length and parameters of this axis pass.
void resize_along_axis_preplanned(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    int axis,
    const LSParams& p,
    const Plan1D& plan);

void resize_along_axis_preplanned_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    int axis,
    const LSParams& p,
    const Plan1D& plan);

void resize_2d_linear(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1);

void resize_2d_linear_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1);

void resize_2d_linear_preplanned(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1,
    const Plan1D& plan0,
    const Plan1D& plan1);

void resize_2d_linear_preplanned_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1,
    const Plan1D& plan0,
    const Plan1D& plan1);

void resize_3d_linear(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1,
    const LSParams& p2);

void resize_3d_linear_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1,
    const LSParams& p2);

void resize_3d_linear_preplanned(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1,
    const LSParams& p2,
    const Plan1D& plan0,
    const Plan1D& plan1,
    const Plan1D& plan2);

void resize_3d_linear_preplanned_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p1,
    const LSParams& p2,
    const Plan1D& plan0,
    const Plan1D& plan1,
    const Plan1D& plan2);

void resize_3d_linear_axis02(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p2);

void resize_3d_linear_axis02_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p2);

void resize_3d_linear_axis02_preplanned(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p2,
    const Plan1D& plan0,
    const Plan1D& plan2);

void resize_3d_linear_axis02_preplanned_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p0,
    const LSParams& p2,
    const Plan1D& plan0,
    const Plan1D& plan2);

void resize_3d_linear_axis12(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p1,
    const LSParams& p2);

void resize_3d_linear_axis12_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p1,
    const LSParams& p2);

void resize_3d_linear_axis12_preplanned(
    const double* LS_RESTRICT in,
    double* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p1,
    const LSParams& p2,
    const Plan1D& plan1,
    const Plan1D& plan2);

void resize_3d_linear_axis12_preplanned_f32(
    const float* LS_RESTRICT in,
    float* LS_RESTRICT out,
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_shape,
    const LSParams& p1,
    const LSParams& p2,
    const Plan1D& plan1,
    const Plan1D& plan2);

} // namespace lsresize
