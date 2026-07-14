#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "resize_1d.h"
#include "resize_nd.h"
#include "utils.h"

using lsresize::LSParams;

namespace {

std::int64_t element_count(const std::vector<std::int64_t>& shape)
{
  std::int64_t total = 1;
  for (const std::int64_t extent : shape) {
    total *= extent;
  }
  return total;
}

struct ResizeResult {
  std::vector<double> values;
  std::vector<std::int64_t> shape;
};

ResizeResult resize_nd(
    const std::vector<double>& data,
    const std::vector<std::int64_t>& input_shape,
    const std::vector<double>& zoom,
    int interpolation_degree,
    int analysis_degree,
    int synthesis_degree)
{
  if (input_shape.size() != zoom.size()) {
    throw std::invalid_argument("zoom rank must match the input rank");
  }

  std::vector<std::int64_t> output_shape = input_shape;
  for (std::size_t axis = 0; axis < input_shape.size(); ++axis) {
    const int output = lsresize::calculate_output_size_1d(
        static_cast<int>(input_shape[axis]),
        zoom[axis]);
    output_shape[axis] = output;
  }

  std::vector<double> current = data;
  std::vector<std::int64_t> current_shape = input_shape;
  for (std::size_t axis = 0; axis < input_shape.size(); ++axis) {
    std::vector<std::int64_t> next_shape = current_shape;
    next_shape[axis] = output_shape[axis];
    std::vector<double> next(
        static_cast<std::size_t>(element_count(next_shape)), 0.0);
    const LSParams params{
        interpolation_degree,
        analysis_degree,
        synthesis_degree,
        zoom[axis],
        0.0};
    lsresize::resize_along_axis(
        current.data(),
        next.data(),
        current_shape,
        next_shape,
        static_cast<int>(axis),
        params);
    current.swap(next);
    current_shape.swap(next_shape);
  }
  return ResizeResult{std::move(current), std::move(current_shape)};
}

bool all_close(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    double tolerance)
{
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) ||
        std::abs(actual[i] - expected[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

bool check(bool condition, const std::string& description)
{
  std::cout << (condition ? "PASS  " : "FAIL  ") << description << '\n';
  return condition;
}

} // namespace

int main()
{
  bool ok = true;

  std::vector<double> signal(10);
  for (std::size_t i = 0; i < signal.size(); ++i) {
    signal[i] = std::sin(0.37 * static_cast<double>(i)) +
                0.1 * static_cast<double>(i);
  }

  // Nominal zoom is a size request only. Requests resolving to the same
  // integer shape must use the exact same endpoint-aligned geometry.
  const ResizeResult half = resize_nd(signal, {10}, {0.50}, 3, 1, 3);
  const ResizeResult near_half = resize_nd(signal, {10}, {0.51}, 3, 1, 3);
  ok &= check(
      half.shape == std::vector<std::int64_t>{5} &&
          half.values == near_half.values,
      "equal realized shapes have identical endpoint geometry");

  // A realized identity is an exact copy when interpolation and synthesis
  // spaces match, including projection configurations.
  const ResizeResult identity = resize_nd(signal, {10}, {1.001}, 3, 1, 3);
  ok &= check(
      identity.shape == std::vector<std::int64_t>{10} &&
          identity.values == signal,
      "projection identity preserves every sample exactly");

  const std::vector<LSParams> configurations = {
      {1, -1, 1, 0.5, 0.0},
      {3, -1, 3, 0.5, 0.0},
      {1, 0, 1, 0.5, 0.0},
      {2, 1, 2, 0.5, 0.0},
      {3, 1, 3, 0.5, 0.0},
      {3, 3, 3, 0.5, 0.0},
  };
  const std::vector<double> constant(13 * 7, 2.75);
  for (const LSParams& params : configurations) {
    const ResizeResult resized = resize_nd(
        constant,
        {13, 7},
        {0.5, 1.4},
        params.interp_degree,
        params.analy_degree,
        params.synthe_degree);
    ok &= check(
        all_close(
            resized.values,
            std::vector<double>(resized.values.size(), 2.75),
            2e-9),
        "constant preservation for degrees (" +
            std::to_string(params.interp_degree) + ", " +
            std::to_string(params.analy_degree) + ", " +
            std::to_string(params.synthe_degree) + ")");
  }

  const ResizeResult replicated = resize_nd({-4.25}, {1}, {7.0}, 3, 1, 3);
  ok &= check(
      replicated.shape == std::vector<std::int64_t>{7} &&
          replicated.values == std::vector<double>(7, -4.25),
      "singleton input is replicated");

  const ResizeResult projected_single =
      resize_nd({0.0, 2.0, 8.0, 10.0}, {4}, {0.01}, 3, 1, 3);
  ok &= check(
      projected_single.shape == std::vector<std::int64_t>{1} &&
          projected_single.values == std::vector<double>{5.0},
      "singleton projection output is the line mean");

  std::cout << (ok ? "All canonical resize checks passed.\n"
                   : "Canonical resize checks failed.\n");
  return ok ? 0 : 1;
}
