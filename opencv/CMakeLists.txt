cmake_minimum_required(VERSION 3.8)
project(opencv)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES
        DisplayImage.cpp
        )

add_executable(DisplayImage
        ${SOURCE_FILES}
        )
target_link_libraries(DisplayImage ${OpenCV_LIBS})
