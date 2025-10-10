# splineops/src/splineops/resize/lsresize/CMakeLists.txt
cmake_minimum_required(VERSION 3.17)
project(lsresize LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LSRESIZE_OPENMP "Enable OpenMP parallelization" ON)
option(LSRESIZE_BUILD_DEMO "Build the run_demo executable" ON)

# ---- library ---------------------------------------------------------------
set(LSR_INC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(lsresize_core STATIC
  src/filters.cpp
  src/resize1d.cpp
  src/resizend.cpp
)

# Header root so that #include "lsresize/resize1d.h" works
target_include_directories(lsresize_core PUBLIC "${LSR_INC}")

# MSVC flags: don't force /O2 in Debug
if (MSVC)
  target_compile_options(lsresize_core PRIVATE
    /EHsc /permissive-
    $<$<CONFIG:Release>:/O2 /DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:/O2 /DNDEBUG>
    $<$<CONFIG:MinSizeRel>:/O1 /DNDEBUG>
  )
else()
  target_compile_options(lsresize_core PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
  )
endif()

if (LSRESIZE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(lsresize_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lsresize_core PUBLIC LSRESIZE_USE_OPENMP=1)
  endif()
endif()

# ---- demo exe --------------------------------------------------------------
if (LSRESIZE_BUILD_DEMO)
  add_executable(run_demo src/run_demo.cpp)
  target_link_libraries(run_demo PRIVATE lsresize_core)
  # extra explicit include for VS; harmless elsewhere
  target_include_directories(run_demo PRIVATE "${LSR_INC}")

  if (MSVC)
    target_compile_options(run_demo PRIVATE
      /EHsc /permissive-
      $<$<CONFIG:Release>:/O2 /DNDEBUG>
      $<$<CONFIG:RelWithDebInfo>:/O2 /DNDEBUG>
      $<$<CONFIG:MinSizeRel>:/O1 /DNDEBUG>
    )
  else()
    target_compile_options(run_demo PRIVATE
      $<$<CONFIG:Release>:-O3 -DNDEBUG>
      $<$<CONFIG:RelWithDebInfo>:-O3 -DNDEBUG>
      $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    )
  endif()
endif()
