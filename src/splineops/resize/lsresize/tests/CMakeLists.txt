# splineops/src/splineops/resize/lsresize/tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.17)
project(lsresize_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pull in the parent project as a subdirectory (keeps core CMake clean)
add_subdirectory(.. lsresize_build)

add_executable(run_demo main.cpp)
target_link_libraries(run_demo PRIVATE lsresize_core)

if (MSVC)
  target_compile_options(run_demo PRIVATE /O2 /DNDEBUG /EHsc /permissive-)
else()
  target_compile_options(run_demo PRIVATE -O3 -DNDEBUG)
endif()
