cmake_minimum_required(VERSION 3.16)
project(CylinderFlow_FNO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# FFTW3 (installed via libfftw3-dev). No FindFFTW3 module ships with CMake,
# so locate header + lib directly.
find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTW3_LIBRARY fftw3 REQUIRED)

add_library(fno
  src/config.cpp
  src/foam_reader.cpp
  src/grid.cpp
  src/fno.cpp
  src/math.cpp
  src/train.cpp
  src/vtk.cpp
  src/weights.cpp
)
target_include_directories(fno PUBLIC src ${FFTW3_INCLUDE_DIR})
target_link_libraries(fno PUBLIC ${FFTW3_LIBRARY})

add_executable(cylfno main.cpp)
target_link_libraries(cylfno PRIVATE fno)
