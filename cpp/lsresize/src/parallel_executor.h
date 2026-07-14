// splineops/cpp/lsresize/src/parallel_executor.h
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace lsresize::detail {

using ParallelTask = std::function<void()>;

// Hard process-wide ceiling for one parallel region, including its submitting
// thread.  The line scheduler applies the same limit to environment overrides,
// and the executor enforces it again before creating persistent workers.
inline constexpr std::size_t kMaxParallelParticipants = 256;

// Run every task on a process-wide persistent executor and wait for completion.
// Exceptions raised by tasks are rethrown in the calling thread after the other
// tasks in the batch have completed.
void run_parallel_tasks(std::vector<ParallelTask> tasks);

// Nested parallel regions execute serially. This avoids a worker waiting for
// work that can only be serviced by the same bounded executor.
bool in_parallel_worker() noexcept;

} // namespace lsresize::detail
