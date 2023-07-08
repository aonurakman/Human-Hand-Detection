cmake_minimum_required(VERSION 3.0)
project(CV_Project)

include(CTest)
find_package(OpenCV REQUIRED)

enable_testing()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(program src/main.cpp src/io.cpp src/error_check.cpp)


target_link_libraries(program ${OpenCV_LIBS})



