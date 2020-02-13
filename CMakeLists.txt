cmake_minimum_required(VERSION 3.15)

project(Example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-copy -fopenmp")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra")

find_package(Eigen3 3.3.9 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(main ceres)

