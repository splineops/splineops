# splineops/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(splineops_native LANGUAGES CXX)

option(LSRESIZE_USE_OPENMP "Enable OpenMP for parallel loops" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
  add_compile_options(/W4 /permissive- /Zc:preprocessor /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Let pybind11 find the matching Python
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if (LSRESIZE_USE_OPENMP)
  find_package(OpenMP QUIET)
endif()

# ---- Absolute source paths (robust on Windows) ----
set(LSRESIZE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lsresize/src")
set(PYBIND_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/cpp/pybind")

add_library(lsresize STATIC
  "${LSRESIZE_SRC_DIR}/filters.cpp"
  "${LSRESIZE_SRC_DIR}/resize1d.cpp"
  "${LSRESIZE_SRC_DIR}/resizend.cpp"
)
target_include_directories(lsresize PUBLIC "${LSRESIZE_SRC_DIR}")
target_compile_features(lsresize PUBLIC cxx_std_17)
if (OpenMP_CXX_FOUND)
  target_link_libraries(lsresize PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(lsresize PUBLIC LSRESIZE_WITH_OPENMP=1)
endif()

pybind11_add_module(_lsresize
  "${PYBIND_DIR}/_lsresize_bindings.cpp"
)
target_link_libraries(_lsresize PRIVATE lsresize)

install(TARGETS _lsresize
        LIBRARY DESTINATION splineops
        RUNTIME DESTINATION splineops
        ARCHIVE DESTINATION splineops)

message(STATUS "OpenMP enabled: " $<BOOL:${OpenMP_CXX_FOUND}>)
