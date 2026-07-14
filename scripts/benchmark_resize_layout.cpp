// Standalone memory-layout experiment for the native resize axis kernels.
//
// It deliberately contains no spline arithmetic.  The goal is to compare the
// existing cache-sized [axis, batch-of-lines] gather/scatter layout with a
// whole-axis-front buffer populated by a tiled tensor transpose.  Compile with:
//
//   c++ -O3 -DNDEBUG -std=c++17 -march=native \
//       scripts/benchmark_resize_layout.cpp -o /tmp/benchmark_resize_layout
//
// The reported scratch sizes are layout-only.  A production implementation
// can reuse an N-D ping-pong buffer for the full axis-front buffer, but cannot
// do that while retaining the current canonical-layout intermediate.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

struct Case {
  const char* name;
  std::vector<int64_t> shape;
  int axis;
};

int64_t product(const std::vector<int64_t>& shape)
{
  return std::accumulate(
      shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

std::vector<int64_t> strides(const std::vector<int64_t>& shape)
{
  std::vector<int64_t> result(shape.size(), 1);
  for (int d = static_cast<int>(shape.size()) - 2; d >= 0; --d) {
    result[static_cast<size_t>(d)] =
        result[static_cast<size_t>(d + 1)] * shape[static_cast<size_t>(d + 1)];
  }
  return result;
}

// The line order matches resize_nd.cpp: all non-axis dimensions are flattened
// in row-major order.
int64_t line_offset(
    int64_t line,
    int axis,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride)
{
  int64_t offset = 0;
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    if (d == axis) {
      continue;
    }
    const int64_t coordinate = line % shape[static_cast<size_t>(d)];
    line /= shape[static_cast<size_t>(d)];
    offset += coordinate * stride[static_cast<size_t>(d)];
  }
  return offset;
}

std::vector<int64_t> line_offsets(
    const std::vector<int64_t>& shape,
    int axis)
{
  const auto stride = strides(shape);
  const int64_t lines = product(shape) / shape[static_cast<size_t>(axis)];
  std::vector<int64_t> result(static_cast<size_t>(lines));
  for (int64_t line = 0; line < lines; ++line) {
    result[static_cast<size_t>(line)] =
        line_offset(line, axis, shape, stride);
  }
  return result;
}

template <typename Function>
double median_ms(Function&& function, int warmups = 1, int repeats = 5)
{
  for (int i = 0; i < warmups; ++i) {
    function();
  }
  std::vector<double> timings;
  timings.reserve(static_cast<size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    const auto begin = Clock::now();
    function();
    const auto end = Clock::now();
    timings.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  std::sort(timings.begin(), timings.end());
  return timings[timings.size() / 2];
}

template <typename Input, typename Work>
void gather_blocked(
    const std::vector<Input>& input,
    std::vector<Work>& block_buffer,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& offsets,
    int axis,
    int batch_lines)
{
  const auto stride = strides(shape);
  const int N = static_cast<int>(shape[static_cast<size_t>(axis)]);
  const int64_t lines = product(shape) / N;
  double checksum = 0.0;
  for (int64_t first = 0; first < lines; first += batch_lines) {
    const int B = static_cast<int>(
        std::min<int64_t>(batch_lines, lines - first));
    for (int n = 0; n < N; ++n) {
      Work* destination = block_buffer.data() + static_cast<size_t>(n) * B;
      const int64_t axis_delta = static_cast<int64_t>(n) * stride[axis];
      for (int b = 0; b < B; ++b) {
        const int64_t offset = offsets[static_cast<size_t>(first + b)];
        destination[b] = static_cast<Work>(input[offset + axis_delta]);
      }
    }
    checksum += static_cast<double>(block_buffer[static_cast<size_t>(N * B - 1)]);
  }
  g_sink += checksum;
}

template <typename Work, typename Output>
void scatter_blocked(
    const std::vector<Work>& block_buffer,
    std::vector<Output>& output,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& offsets,
    int axis,
    int batch_lines)
{
  const auto stride = strides(shape);
  const int N = static_cast<int>(shape[static_cast<size_t>(axis)]);
  const int64_t lines = product(shape) / N;
  for (int64_t first = 0; first < lines; first += batch_lines) {
    const int B = static_cast<int>(
        std::min<int64_t>(batch_lines, lines - first));
    for (int b = 0; b < B; ++b) {
      const int64_t offset = offsets[static_cast<size_t>(first + b)];
      for (int n = 0; n < N; ++n) {
        output[offset + static_cast<int64_t>(n) * stride[axis]] =
            static_cast<Output>(block_buffer[static_cast<size_t>(n) * B + b]);
      }
    }
  }
  g_sink += static_cast<double>(output.back());
}

// Populate an axis-front [N, lines] buffer.  The two inner-loop orders expose
// the source-contiguous versus destination-contiguous tradeoff; tiling bounds
// the strided side's active cache footprint.
template <bool SourceContiguous, typename Input, typename Work>
void gather_axis_front_tiled(
    const std::vector<Input>& input,
    std::vector<Work>& axis_front,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& offsets,
    int axis,
    int tile)
{
  const auto stride = strides(shape);
  const int N = static_cast<int>(shape[static_cast<size_t>(axis)]);
  const int64_t lines = product(shape) / N;
  for (int64_t line0 = 0; line0 < lines; line0 += tile) {
    const int64_t line1 = std::min<int64_t>(line0 + tile, lines);
    for (int n0 = 0; n0 < N; n0 += tile) {
      const int n1 = std::min(n0 + tile, N);
      if constexpr (SourceContiguous) {
        for (int64_t line = line0; line < line1; ++line) {
          const int64_t offset = offsets[static_cast<size_t>(line)];
          for (int n = n0; n < n1; ++n) {
            axis_front[static_cast<size_t>(n) * lines + line] =
                static_cast<Work>(
                    input[offset + static_cast<int64_t>(n) * stride[axis]]);
          }
        }
      } else {
        for (int n = n0; n < n1; ++n) {
          const int64_t axis_delta = static_cast<int64_t>(n) * stride[axis];
          Work* destination = axis_front.data() + static_cast<size_t>(n) * lines;
          for (int64_t line = line0; line < line1; ++line) {
            const int64_t offset = offsets[static_cast<size_t>(line)];
            destination[line] = static_cast<Work>(input[offset + axis_delta]);
          }
        }
      }
    }
  }
  g_sink += static_cast<double>(axis_front.back());
}

template <bool DestinationContiguous, typename Work, typename Output>
void scatter_axis_front_tiled(
    const std::vector<Work>& axis_front,
    std::vector<Output>& output,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& offsets,
    int axis,
    int tile)
{
  const auto stride = strides(shape);
  const int N = static_cast<int>(shape[static_cast<size_t>(axis)]);
  const int64_t lines = product(shape) / N;
  for (int64_t line0 = 0; line0 < lines; line0 += tile) {
    const int64_t line1 = std::min<int64_t>(line0 + tile, lines);
    for (int n0 = 0; n0 < N; n0 += tile) {
      const int n1 = std::min(n0 + tile, N);
      if constexpr (DestinationContiguous) {
        for (int64_t line = line0; line < line1; ++line) {
          const int64_t offset = offsets[static_cast<size_t>(line)];
          for (int n = n0; n < n1; ++n) {
            output[offset + static_cast<int64_t>(n) * stride[axis]] =
                static_cast<Output>(
                    axis_front[static_cast<size_t>(n) * lines + line]);
          }
        }
      } else {
        for (int n = n0; n < n1; ++n) {
          const int64_t axis_delta = static_cast<int64_t>(n) * stride[axis];
          const Work* source =
              axis_front.data() + static_cast<size_t>(n) * lines;
          for (int64_t line = line0; line < line1; ++line) {
            const int64_t offset = offsets[static_cast<size_t>(line)];
            output[offset + axis_delta] = static_cast<Output>(source[line]);
          }
        }
      }
    }
  }
  g_sink += static_cast<double>(output.back());
}

template <typename Input, typename Work>
void run_case(const Case& benchmark, const char* conversion)
{
  const int64_t input_elements = product(benchmark.shape);
  const int N = static_cast<int>(benchmark.shape[benchmark.axis]);
  std::vector<int64_t> output_shape = benchmark.shape;
  output_shape[benchmark.axis] = std::max<int64_t>(1, N / 2);
  const int outN = static_cast<int>(output_shape[benchmark.axis]);
  const int64_t output_elements = product(output_shape);
  const int64_t lines = input_elements / N;

  std::vector<Input> input(static_cast<size_t>(input_elements));
  for (int64_t i = 0; i < input_elements; ++i) {
    input[static_cast<size_t>(i)] = static_cast<Input>(
        std::sin(static_cast<double>(i) * 0.001));
  }
  std::vector<Input> output(static_cast<size_t>(output_elements));
  const auto input_offsets = line_offsets(benchmark.shape, benchmark.axis);
  const auto output_offsets = line_offsets(output_shape, benchmark.axis);

  std::cout << std::fixed << std::setprecision(3);
  for (const int batch : {32, 64, 96, 128, 256}) {
    std::vector<Work> gather_buffer(static_cast<size_t>(N) * batch);
    std::vector<Work> scatter_buffer(static_cast<size_t>(outN) * batch, Work{0.25});
    const double gather_ms = median_ms([&] {
      gather_blocked(
          input, gather_buffer, benchmark.shape, input_offsets,
          benchmark.axis, batch);
    });
    const double scatter_ms = median_ms([&] {
      scatter_blocked(
          scatter_buffer, output, output_shape, output_offsets,
          benchmark.axis, batch);
    });
    const uint64_t scratch = static_cast<uint64_t>(batch) *
        static_cast<uint64_t>(N + outN) * sizeof(Work);
    std::cout << benchmark.name << ',' << conversion << ",blocked," << batch
              << ',' << gather_ms << ',' << scatter_ms << ','
              << gather_ms + scatter_ms << ',' << scratch << '\n';
  }

  std::vector<Work> input_axis_front(static_cast<size_t>(input_elements));
  std::vector<Work> output_axis_front(
      static_cast<size_t>(output_elements), Work{0.25});
  for (const int tile : {16, 32, 64, 128}) {
    const double gather_source_ms = median_ms([&] {
      gather_axis_front_tiled<true>(
          input, input_axis_front, benchmark.shape, input_offsets,
          benchmark.axis, tile);
    });
    const double gather_destination_ms = median_ms([&] {
      gather_axis_front_tiled<false>(
          input, input_axis_front, benchmark.shape, input_offsets,
          benchmark.axis, tile);
    });
    const double scatter_destination_ms = median_ms([&] {
      scatter_axis_front_tiled<true>(
          output_axis_front, output, output_shape, output_offsets,
          benchmark.axis, tile);
    });
    const double scatter_source_ms = median_ms([&] {
      scatter_axis_front_tiled<false>(
          output_axis_front, output, output_shape, output_offsets,
          benchmark.axis, tile);
    });
    const double gather_ms = std::min(gather_source_ms, gather_destination_ms);
    const double scatter_ms = std::min(scatter_destination_ms, scatter_source_ms);
    const uint64_t scratch = static_cast<uint64_t>(input_elements) * sizeof(Work) +
        static_cast<uint64_t>(output_elements) * sizeof(Work);
    std::cout << benchmark.name << ',' << conversion << ",axis_front," << tile
              << ',' << gather_ms << ',' << scatter_ms << ','
              << gather_ms + scatter_ms << ',' << scratch << '\n';
  }
}

} // namespace

int main()
{
  const std::vector<Case> cases = {
      {"2d_2048_axis0", {2048, 2048}, 0},
      {"2d_2048_axis1", {2048, 2048}, 1},
      {"3d_256x256x64_axis0", {256, 256, 64}, 0},
      {"3d_256x256x64_axis1", {256, 256, 64}, 1},
      {"3d_256x256x64_axis2", {256, 256, 64}, 2},
  };

  std::cout << "case,conversion,layout,tile_or_batch,gather_ms,scatter_ms,total_ms,scratch_bytes\n";
  for (const Case& benchmark : cases) {
    run_case<float, float>(benchmark, "f32_to_f32");
    run_case<float, double>(benchmark, "f32_to_f64");
    run_case<double, double>(benchmark, "f64_to_f64");
  }
  std::cerr << "checksum=" << g_sink << '\n';
  return EXIT_SUCCESS;
}
