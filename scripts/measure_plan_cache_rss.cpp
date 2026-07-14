// Standalone retained-memory probe for the native Plan1D cache.
//
// Build from the repository root with:
//   c++ -std=c++17 -O2 -pthread -Icpp/lsresize/src cpp/lsresize/src/filters.cpp cpp/lsresize/src/parallel_executor.cpp cpp/lsresize/src/profile_utils.cpp cpp/lsresize/src/resize_1d.cpp scripts/measure_plan_cache_rss.cpp -o /tmp/measure_plan_cache_rss

#include "resize_1d.h"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

std::size_t resident_bytes()
{
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  std::size_t total_pages = 0;
  std::size_t resident_pages = 0;
  statm >> total_pages >> resident_pages;
  (void)total_pages;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0
      ? resident_pages * static_cast<std::size_t>(page_size)
      : 0;
#else
  return 0;
#endif
}

} // namespace

int main()
{
#if defined(_WIN32)
  _putenv_s("LSRESIZE_PLAN_CACHE_SIZE", "32");
  _putenv_s("LSRESIZE_PLAN_CACHE_BYTES", "");
#else
  ::setenv("LSRESIZE_PLAN_CACHE_SIZE", "32", 1);
  ::unsetenv("LSRESIZE_PLAN_CACHE_BYTES");
#endif

  const std::size_t before = resident_bytes();
  const lsresize::LSParams params{3, 3, 3, 0.37, 0.0};
  for (int index = 0; index < 32; ++index) {
    (void)lsresize::get_plan_1d_cached(100000 + index, params);
  }
  const std::size_t after = resident_bytes();

  constexpr double mib = 1024.0 * 1024.0;
  std::cout << "rss_before_mib=" << static_cast<double>(before) / mib << '\n'
            << "rss_after_mib=" << static_cast<double>(after) / mib << '\n'
            << "rss_delta_mib="
            << static_cast<double>(after - before) / mib << '\n';
  return 0;
}
