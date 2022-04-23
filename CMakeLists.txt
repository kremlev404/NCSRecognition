cmake_minimum_required(VERSION 3.13)

project(NCSRecognition)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

execute_process(COMMAND
        dpkg-architecture
        -qDEB_HOST_ARCH
        OUTPUT_VARIABLE
        CMAKE_DEB_HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

message("-- [MAIN] ARCH IS: ${CMAKE_DEB_HOST_ARCH}")

set(CMAKE_CXX_STANDARD 20)
if (${CMAKE_DEB_HOST_ARCH} MATCHES "arm")
    set(IS_RASPBERRY ON)
    find_library(WIRING_PI_LIB wiringPi)
    message("-- [MAIN] Found WIRING_PI_LIB ${WIRING_PI_LIB}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -std=c++17 -Wno-psabi -pthread")
elseif (${CMAKE_DEB_HOST_ARCH} MATCHES "64")
    set(IS_RASPBERRY OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wno-psabi -pthread")
endif ()

find_package(OpenCV REQUIRED)
set(REQUIRED_OpenCV_VERSION 4.1)
if (OpenCV_VERSION VERSION_LESS REQUIRED_OpenCV_VERSION)
    message("-- [MAIN] Error: At least OpenCV ${REQUIRED_OpenCV_VERSION} is required")
endif ()
message("-- [MAIN] Found OpenCV ${OpenCV_VERSION}")

set(IE_DIR $ENV{INTEL_OPENVINO_DIR}/inference_engine/)
message("-- [MAIN] IE_DIR: ${IE_DIR}")

include_directories("include")
include_directories("3rdparty_include")

add_subdirectory(src)