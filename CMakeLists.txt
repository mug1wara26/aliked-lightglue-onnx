cmake_minimum_required(VERSION 2.8)
project(aliked_lightglue)

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()
MESSAGE("build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -fopenmp -march=native -pthread")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(USE_TENSORRT ON)
set(USE_TENSORFLOW OFF)
set(USE_OPENVINO OFF)

# use the default opencv in ubuntu 20.04
# set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4 QUIET)
message(STATUS "OpenCV version: " ${OpenCV_VERSION})

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

include_directories(
include
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    include/ExtractorType.h
    include/Aliked/IOnnxRunner.h
    include/Aliked/Config.h
    include/Aliked/AlikedOnnxManager.h
    include/Aliked/OnnxRunner.h
    include/Lightglue/ILightglueManager.h
    include/Lightglue/LightglueConf.h
    include/Lightglue/LightglueOnnxManager.h
    include/Lightglue/OnnxRunner.h
    src/OnnxRunner/Aliked.cpp
    src/OnnxRunner/Lightglue.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
)

if(${USE_TENSORRT})
    ADD_DEFINITIONS(-DUSE_TENSORRT)
    ADD_DEFINITIONS(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser ${CUDA_LIBRARIES})
endif()

set(ONNXRUNTIME_DIR "/home/lawrence/aloysius_code/onnxruntime-linux-x64-gpu-1.16.3")

include_directories(${ONNXRUNTIME_DIR}/include)

set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.16.3)
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBS})

add_executable(test_aliked
        examples/test_aliked.cpp)
target_link_libraries(test_aliked ${PROJECT_NAME})

add_executable(lightglue
        examples/aliked_lightglue.cpp)
target_link_libraries(lightglue ${PROJECT_NAME})
