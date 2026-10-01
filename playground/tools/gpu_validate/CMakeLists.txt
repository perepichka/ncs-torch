# GPU capability validator for the NCS playground (raylib 6.0, OpenGL 4.3 backend).
#   cmake -S . -B build [-DPG_WITH_CUDA=ON] && cmake --build build --config Release
#   build/Release/gpu_validate.exe   (Windows)   |   build/gpu_validate   (Linux)
cmake_minimum_required(VERSION 3.25)
project(pg_gpu_validate CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(OPENGL_VERSION "4.3" CACHE STRING "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(raylib GIT_REPOSITORY https://github.com/raysan5/raylib.git GIT_TAG 6.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(raylib)

option(PG_WITH_CUDA "Build the CUDA-OpenGL interop test (needs CUDA Toolkit)" OFF)

add_executable(gpu_validate validate.cpp)
target_link_libraries(gpu_validate PRIVATE raylib)
if(PG_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(gpu_validate PRIVATE cuda_interop.cpp)
  target_compile_definitions(gpu_validate PRIVATE PG_WITH_CUDA)
  target_link_libraries(gpu_validate PRIVATE CUDA::cudart)
endif()
