# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT(DvoGpu C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#==============================================================================
# Dependencies
#==============================================================================

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_61 -std=c++11 -O1)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v --use_fast_math --compiler-options -Wall)

# include sophus and Eigen
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/third_party/include/ )

# Eigen
SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# ------------------------------------------------------------------------
# OpenCV
#FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)
#find_package(OpenCV REQUIRED)
#include_directories( ${OpenCV_INCLUDE_DIRS} )

FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS
    core highgui imgproc video
    REQUIRED
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


## OpenNI2
FIND_PACKAGE(OpenNI2 REQUIRED )
INCLUDE_DIRECTORIES(${OPENNI2_INCLUDE_DIRS})


#==============================================================================
# Application
#==============================================================================

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O1 -funroll-loops" )

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_SOURCE_DIR=${PROJECT_SOURCE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

# applications
#FILE(GLOB incs "src/*.cuh")
#FILE(GLOB srcs "src/*.cu")

## ADD_EXECUTABLE(dvo_cpu ${incs} ${srcs} )
#CUDA_ADD_EXECUTABLE(dvo_gpu
#	${incs}
#	${srcs}
#)

CUDA_ADD_EXECUTABLE(dvo_gpu
        src/dvo_cuda_kernels.cuh
        src/dvo.cuh src/dvo.cu
        src/helper.cuh src/helper.cu
        src/tum_benchmark.cuh
        src/main.cu
)
CUDA_ADD_CUBLAS_TO_TARGET(dvo_gpu)
TARGET_LINK_LIBRARIES(dvo_gpu ${OpenCV_LIBS} ${OPENNI2_LIBRARY})
