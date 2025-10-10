# splineops/src/splineops/resize/lsresize/CMakeLists.txt
cmake_minimum_required(VERSION 3.17)
project(lsresize LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LSRESIZE_OPENMP "Enable OpenMP parallelization" ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
if(LSRESIZE_OPENMP)
  find_package(OpenMP)
endif()

add_library(lsresize MODULE
  src/filters.cpp
  src/resize1d.cpp
  src/resizend.cpp
  src/bindings.cpp
)
target_include_directories(lsresize PRIVATE include)

if(LSRESIZE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(lsresize PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(lsresize PRIVATE LSRESIZE_USE_OPENMP=1)
endif()

target_compile_options(lsresize PRIVATE -O3 -DNDEBUG)
target_link_libraries(lsresize PRIVATE pybind11::module)

# correct Python extension suffix (.pyd on Windows)
set_target_properties(lsresize PROPERTIES PREFIX "" SUFFIX "${Python3_EXTENSION_SUFFIX}")
