# splineops/cpp/lsresize/src/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(lsresize_demo LANGUAGES CXX)

# Parallel options (same semantics as top-level)
option(LSRESIZE_USE_OPENMP  "Enable OpenMP for parallel loops" ON)
option(LSRESIZE_USE_STDTHREAD
       "Use std::thread for parallel loops if OpenMP is unavailable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source dir is the subfolder next to this CMakeLists
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
message(STATUS "SRC_DIR = ${SRC_DIR}") # helpful sanity check

# Core library
add_library(lsresize STATIC
  "${SRC_DIR}/filters.cpp"
  "${SRC_DIR}/resize1d.cpp"
  "${SRC_DIR}/resizend.cpp"
)
target_include_directories(lsresize PUBLIC "${SRC_DIR}")
target_compile_features(lsresize PUBLIC cxx_std_17)

# Warnings
if (MSVC)
  target_compile_options(lsresize PRIVATE /W4 /permissive- /Zc:preprocessor /Zc:__cplusplus)
else()
  target_compile_options(lsresize PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Parallel backend selection (OpenMP or std::thread) ---

if (LSRESIZE_USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

if (OpenMP_CXX_FOUND)
  message(STATUS "lsresize demo: building with OpenMP")
  target_link_libraries(lsresize PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(lsresize PUBLIC LSRESIZE_WITH_OPENMP=1)
elseif (LSRESIZE_USE_STDTHREAD)
  message(STATUS "lsresize demo: OpenMP not found, using std::thread fallback")
  target_compile_definitions(lsresize PUBLIC LSRESIZE_WITH_STDTHREAD=1)
else()
  message(STATUS "lsresize demo: building single-threaded (no OpenMP, no std::thread)")
endif()

# Demo executable (placed at top-level next to this CMakeLists)
# If you DIDN'T move it, change to "${SRC_DIR}/run_demo.cpp" instead.
add_executable(run_demo "${CMAKE_CURRENT_SOURCE_DIR}/run_demo.cpp")
target_link_libraries(run_demo PRIVATE lsresize)

# Simple test hook
include(CTest)
if (BUILD_TESTING)
  add_test(NAME run_demo_ok COMMAND run_demo)
endif()

# Usage (example):
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#   .\build\Release\run_demo.exe
