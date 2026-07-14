// splineops/cpp/lsresize/src/profile_utils.h
#pragma once

#include <chrono>
#include <cstdint>

namespace lsresize {
namespace profile {

enum class Phase : int {
  PlanBuild = 0,
  NdAxisTotal,
  LinearDirectTotal,
  Fused2DLinearTotal,
  Fused3DLinearTotal,
  BatchedInterpTotal,
  BatchedInterpGather,
  BatchedInterpPrefilter,
  BatchedInterpAccumulate,
  BatchedInterpScatter,
  BatchedInterpAccumulateScatter,
  BatchedProjectionTotal,
  BatchedProjectionGather,
  BatchedProjectionDcCenter,
  BatchedProjectionInputPrefilter,
  BatchedProjectionIntegrate,
  BatchedProjectionAccumulate,
  BatchedProjectionDiff,
  BatchedProjectionOutputPrefilter,
  BatchedProjectionSampling,
  BatchedProjectionScatter,
  LineFallbackTotal,
  LineFallbackOffset,
  LineFallbackGather,
  LineFallbackResize1D,
  LineFallbackScatter,
  Pipeline1DTotal,
  Pipeline1DRawCopy,
  Pipeline1DPrefilter,
  Pipeline1DIntegrate,
  Pipeline1DExtend,
  Pipeline1DAccumulate,
  Pipeline1DDiff,
  Pipeline1DOutputPrefilter,
  Pipeline1DSampling,
  Pipeline1DOutputCopy,
  Count
};

bool enabled() noexcept;
void add(Phase phase, std::uint64_t elapsed_ns) noexcept;
void print_summary();

class Scope {
public:
  explicit Scope(Phase phase);
  ~Scope() noexcept;

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

private:
  Phase phase_;
  bool active_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace profile
} // namespace lsresize

#define LSRESIZE_PROFILE_CONCAT_IMPL(a, b) a##b
#define LSRESIZE_PROFILE_CONCAT(a, b) LSRESIZE_PROFILE_CONCAT_IMPL(a, b)
#define LSRESIZE_PROFILE_SCOPE(phase) \
  ::lsresize::profile::Scope LSRESIZE_PROFILE_CONCAT(_lsresize_profile_scope_, __LINE__)(phase)
