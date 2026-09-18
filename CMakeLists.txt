cmake_minimum_required(VERSION 3.16)
project(CylinderFlow_FNO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Scaffold: std-only, no external deps so it builds now.
# Next steps (TODO):
#   find_package(Eigen3 REQUIRED)
#   find_package(FFTW3 REQUIRED)  # -> target_link_libraries(... fftw3)
#   target_link_libraries(fno PUBLIC Eigen3::Eigen fftw3)

add_library(fno
  src/config.cpp
  src/foam_reader.cpp
  src/grid.cpp
  src/fno.cpp
  src/train.cpp
)
target_include_directories(fno PUBLIC src)

add_executable(cylfno main.cpp)
target_link_libraries(cylfno PRIVATE fno)
