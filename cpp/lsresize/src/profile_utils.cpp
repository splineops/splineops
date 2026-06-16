// splineops/cpp/lsresize/src/profile_utils.cpp
#include "profile_utils.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace lsresize {
namespace profile {

namespace {

constexpr int kPhaseCount = static_cast<int>(Phase::Count);

struct ThreadCounters {
  std::array<std::uint64_t, kPhaseCount> ns{};
  std::array<std::uint64_t, kPhaseCount> calls{};
};

std::mutex& registry_mutex()
{
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::vector<ThreadCounters*>& registry()
{
  static auto* counters = new std::vector<ThreadCounters*>();
  return *counters;
}

ThreadCounters& thread_counters()
{
  thread_local ThreadCounters* counters = [] {
    auto* ptr = new ThreadCounters();
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().push_back(ptr);
    return ptr;
  }();
  return *counters;
}

bool env_truthy(const char* value)
{
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  if (value[0] == '0') {
    return false;
  }
  if ((value[0] == 'o' || value[0] == 'O') &&
      (value[1] == 'f' || value[1] == 'F') &&
      (value[2] == 'f' || value[2] == 'F') &&
      value[3] == '\0') {
    return false;
  }
  return true;
}

const char* phase_name(Phase phase)
{
  switch (phase) {
    case Phase::PlanBuild: return "plan.build";
    case Phase::NdAxisTotal: return "nd.axis.total";
    case Phase::LinearDirectTotal: return "linear.direct.total";
    case Phase::Fused2DLinearTotal: return "fused2d.linear.total";
    case Phase::Fused3DLinearTotal: return "fused3d.linear.total";
    case Phase::BatchedInterpTotal: return "batched.interp.total";
    case Phase::BatchedInterpGather: return "batched.interp.gather";
    case Phase::BatchedInterpPrefilter: return "batched.interp.prefilter";
    case Phase::BatchedInterpAccumulate: return "batched.interp.accumulate";
    case Phase::BatchedInterpScatter: return "batched.interp.scatter";
    case Phase::BatchedInterpAccumulateScatter: return "batched.interp.accumulate_scatter";
    case Phase::BatchedProjectionTotal: return "batched.projection.total";
    case Phase::BatchedProjectionGather: return "batched.projection.gather";
    case Phase::BatchedProjectionDcCenter: return "batched.projection.dc_center";
    case Phase::BatchedProjectionInputPrefilter: return "batched.projection.input_prefilter";
    case Phase::BatchedProjectionIntegrate: return "batched.projection.integrate";
    case Phase::BatchedProjectionAccumulate: return "batched.projection.accumulate";
    case Phase::BatchedProjectionDiff: return "batched.projection.diff";
    case Phase::BatchedProjectionOutputPrefilter: return "batched.projection.output_prefilter";
    case Phase::BatchedProjectionSampling: return "batched.projection.sampling";
    case Phase::BatchedProjectionScatter: return "batched.projection.scatter";
    case Phase::LineFallbackTotal: return "line_fallback.total";
    case Phase::LineFallbackOffset: return "line_fallback.offset";
    case Phase::LineFallbackGather: return "line_fallback.gather";
    case Phase::LineFallbackResize1D: return "line_fallback.resize1d";
    case Phase::LineFallbackScatter: return "line_fallback.scatter";
    case Phase::Pipeline1DTotal: return "pipeline1d.total";
    case Phase::Pipeline1DRawCopy: return "pipeline1d.raw_copy";
    case Phase::Pipeline1DPrefilter: return "pipeline1d.prefilter";
    case Phase::Pipeline1DIntegrate: return "pipeline1d.integrate";
    case Phase::Pipeline1DExtend: return "pipeline1d.extend";
    case Phase::Pipeline1DAccumulate: return "pipeline1d.accumulate";
    case Phase::Pipeline1DDiff: return "pipeline1d.diff";
    case Phase::Pipeline1DOutputPrefilter: return "pipeline1d.output_prefilter";
    case Phase::Pipeline1DSampling: return "pipeline1d.sampling";
    case Phase::Pipeline1DOutputCopy: return "pipeline1d.output_copy";
    case Phase::Count: return "<count>";
  }
  return "<unknown>";
}

void ensure_atexit_registered()
{
  static const bool registered = [] {
    std::atexit(print_summary);
    return true;
  }();
  (void)registered;
}

} // namespace

bool enabled()
{
  static const bool value = env_truthy(std::getenv("LSRESIZE_PROFILE"));
  if (value) {
    ensure_atexit_registered();
  }
  return value;
}

void add(Phase phase, std::uint64_t elapsed_ns)
{
  if (phase == Phase::Count) {
    return;
  }
  ThreadCounters& counters = thread_counters();
  const int index = static_cast<int>(phase);
  counters.ns[static_cast<size_t>(index)] += elapsed_ns;
  counters.calls[static_cast<size_t>(index)] += 1;
}

Scope::Scope(Phase phase)
  : phase_(phase),
    active_(enabled()),
    start_(active_ ? std::chrono::steady_clock::now()
                   : std::chrono::steady_clock::time_point{})
{}

Scope::~Scope()
{
  if (!active_) {
    return;
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end - start_);
  add(phase_, static_cast<std::uint64_t>(elapsed.count()));
}

void print_summary()
{
  if (!enabled()) {
    return;
  }

  std::array<std::uint64_t, kPhaseCount> ns{};
  std::array<std::uint64_t, kPhaseCount> calls{};
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (const ThreadCounters* counters : registry()) {
      if (counters == nullptr) {
        continue;
      }
      for (int i = 0; i < kPhaseCount; ++i) {
        ns[static_cast<size_t>(i)] += counters->ns[static_cast<size_t>(i)];
        calls[static_cast<size_t>(i)] += counters->calls[static_cast<size_t>(i)];
      }
    }
  }

  std::uint64_t denominator = ns[static_cast<size_t>(Phase::NdAxisTotal)];
  if (denominator == 0) {
    denominator = *std::max_element(ns.begin(), ns.end());
  }
  if (denominator == 0) {
    return;
  }

  std::vector<int> order;
  order.reserve(static_cast<size_t>(kPhaseCount));
  for (int i = 0; i < kPhaseCount; ++i) {
    if (calls[static_cast<size_t>(i)] > 0) {
      order.push_back(i);
    }
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return ns[static_cast<size_t>(a)] > ns[static_cast<size_t>(b)];
  });

  const char* label = std::getenv("LSRESIZE_PROFILE_LABEL");
  const char* threads = std::getenv("LSRESIZE_NUM_THREADS");
  std::cerr << "LSRESIZE_PROFILE_SUMMARY";
  if (label != nullptr && label[0] != '\0') {
    std::cerr << " label=" << label;
  }
  std::cerr << " threads=" << ((threads != nullptr && threads[0] != '\0') ? threads : "<default>");
  std::cerr << " denominator_ms=" << std::fixed << std::setprecision(3)
            << static_cast<double>(denominator) / 1.0e6 << "\n";
  std::cerr << "phase,calls,total_ms,avg_us,pct_of_nd_axis\n";
  for (int i : order) {
    const auto phase = static_cast<Phase>(i);
    const double total_ms = static_cast<double>(ns[static_cast<size_t>(i)]) / 1.0e6;
    const double avg_us =
        static_cast<double>(ns[static_cast<size_t>(i)]) /
        static_cast<double>(calls[static_cast<size_t>(i)]) / 1.0e3;
    const double pct =
        100.0 * static_cast<double>(ns[static_cast<size_t>(i)]) /
        static_cast<double>(denominator);
    std::cerr << phase_name(phase) << ","
              << calls[static_cast<size_t>(i)] << ","
              << std::fixed << std::setprecision(3) << total_ms << ","
              << std::fixed << std::setprecision(3) << avg_us << ","
              << std::fixed << std::setprecision(2) << pct << "\n";
  }
}

} // namespace profile
} // namespace lsresize
