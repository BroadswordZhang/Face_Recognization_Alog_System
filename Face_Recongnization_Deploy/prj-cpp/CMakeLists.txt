#1.cmake verson
cmake_minimum_required(VERSION 2.8)


#2.project name
project(demo C CXX)

#opencv4.1.0
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4.include
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/common/)
include_directories(/usr/local/cuda/include/)

link_directories(/usr/local/cuda/lib64/)

#5.source directory
file(GLOB TEST_SRC
        ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/common/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})


add_executable(demo ${TEST_COMPILE_CODE})

target_link_libraries(demo
        ${OpenCV_LIBS}
        pthread

        cudart
        cudnn
        nvinfer
        nvparsers
        nvonnxparser
        )


# ifeq ($(TRT_STATIC), 1)
#  CUDART_LIB = -lcudart_static
#  CUDNN_LIB = -lcudnn_static
#  CUBLAS_LIB = -lcublas_static
#  MYELIN_LIB = -lmyelin_compiler_static -lmyelin_executor_static -lmyelin_pattern_library_static -lmyelin_pattern_runtime_static
#  NVINFER_LIB = -lnvinfer_static
#  NVPARSERS_LIB = -lnvparsers_static
#  NVINFER_PLUGIN_LIB = -lnvinfer_plugin_static
#  NVONNXPARSERS_LIB = -lnvonnxparser_static
#  NVRTC_LIB = -lnvrtc_static
#  PROTO_LIB = -lprotobuf -lonnx_proto
#  STUBS_DIR = -L"$(TRT_LIB_DIR)/stubs" -Wl,-rpath-link="$(TRT_LIB_DIR)/stubs"
#else
#  CUDART_LIB = -lcudart
#  CUDNN_LIB = -lcudnn
#  CUBLAS_LIB = -lcublas
#  MYELIN_LIB = -lmyelin
#  NVINFER_LIB = -lnvinfer
#  NVPARSERS_LIB = -lnvparsers
#  NVINFER_PLUGIN_LIB = -lnvinfer_plugin
#  NVONNXPARSERS_LIB = -lnvonnxparser
#  NVRTC_LIB = -lnvrtc
#  PROTO_LIBDIR =
#  STUBS_DIR =
#endif