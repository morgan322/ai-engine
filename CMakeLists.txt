cmake_minimum_required(VERSION 3.5)
project(adk C CXX)

set(ADK_VERSION_MAJOR 0)
set(ADK_VERSION_MINOR 1)
set(ADK_VERSION_PATCH 0)
set(ADK_VERSION "${ADK_VERSION_MAJOR}.${ADK_VERSION_MINOR}.${ADK_VERSION_PATCH}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/adk_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/adk_config.h)
install(FILES ADK_config.h DESTINATION include)

include_directories("${CMAKE_SOURCE_DIR}/src")


set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED glib-2.0)
include_directories(${GLIB2_INCLUDE_DIRS})
link_directories(${GLIB2_LIBRARY_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/)
file(GLOB_RECURSE sources src/model/*.hpp src/model/*.cpp src/model/*.h src/model/*.cu )
add_library(engine SHARED ${sources})
target_include_directories(engine PRIVATE include/ src/model/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(engine nvinfer nvinfer_plugin nvparsers "stdc++fs")

add_executable(${PROJECT_NAME} src/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE src/model/ include/ )
target_link_libraries(${PROJECT_NAME} engine ${OpenCV_LIBS} ${GLIB2_LIBRARIES})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers)

add_subdirectory(3rdparty/googletest)
add_subdirectory(unitest)

# add_subdirectory(3rdparty/libyuv)
# add_subdirectory(examples)