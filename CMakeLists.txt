# Build voxblox without ROS

CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(voxblox VERSION 0.1.0 DESCRIPTION "voxblox")

# This should not be required, if not set hopefully the system has the dependencies
#if("${MULTIVIEW_DEPS_DIR}" STREQUAL "")
#   message(FATAL_ERROR "Set MULTIVIEW_DEPS_DIR as the path to a conda env having the dependencies.")
#endif()

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Setting build type: ${CMAKE_BUILD_TYPE}")

# Where to search for dependencies
set(CMAKE_PREFIX_PATH "${MULTIVIEW_DEPS_DIR};${CMAKE_PREFIX_PATH}")
set(CMAKE_MODULE_PATH "${MULTIVIEW_DEPS_DIR}/lib/cmake/Ceres;${CMAKE_MODULE_PATH}")

# Link flags
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${MULTIVIEW_DEPS_DIR}/lib")

# Eigen
find_package(Eigen3 3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
   message("-- Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
else (EIGEN3_FOUND)
   message(FATAL_ERROR "Cannot find Eigen3.")
endif (EIGEN3_FOUND)
include_directories("${EIGEN3_INCLUDE_DIR}")

# GFlags 
find_package(Gflags REQUIRED)
if (Gflags_FOUND)
   message("-- Found Google Flags: ${GFLAGS_INCLUDE_DIRS} in namespace: ${GFLAGS_NAMESPACE}")
else (Gflags_FOUND)
   message(FATAL_ERROR "Can't find Google FLAGS.")
endif (Gflags_FOUND)

# GLog
#add_definitions(-DGLOG_NO_EXPORT)
add_definitions(-DGLOG_USE_GLOG_EXPORT) # must include this for dynamic linking
find_package(Glog REQUIRED)
if (Glog_FOUND)
  message("-- Found Google Logging: ${GLOG_INCLUDE_DIRS}")
else (Glog_FOUND)
  message(FATAL_ERROR "Can't find Glog. Please set GLOG_INCLUDE_DIR & "
    "GLOG_LIBRARY")
endif (Glog_FOUND)

# Boost
find_package(Boost REQUIRED COMPONENTS filesystem)

# PCL
find_package(PCL REQUIRED)
include_directories("${PCL_COMMON_INCLUDE_DIR}")

#find_package(catkin_simple REQUIRED)
#catkin_simple()

set(CMAKE_MACOSX_RPATH 0)
add_definitions(-std=c++17 -Wall -Wextra) # For Protobuf

include_directories("${CMAKE_SOURCE_DIR}/include;${Protobuf_INCLUDE_DIR}")

# minkindr will be cloned as a submodule
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/minkindr/minkindr/include")

add_subdirectory(voxblox)
