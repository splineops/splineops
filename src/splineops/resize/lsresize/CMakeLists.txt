# splineops/src/splineops/resize/lsresize/CMakeLists.txt
cmake_minimum_required(VERSION 3.17)
project(lsresize_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(lsresize_core STATIC
  src/filters.cpp
  src/resize1d.cpp
  src/resizend.cpp
)

target_include_directories(lsresize_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
  target_compile_options(lsresize_core PRIVATE /O2 /DNDEBUG /EHsc /permissive-)
else()
  target_compile_options(lsresize_core PRIVATE -O3 -DNDEBUG)
endif()

# Optional parallelism (harmless if not found)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(lsresize_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(lsresize_core PUBLIC LSRESIZE_USE_OPENMP=1)
endif()
