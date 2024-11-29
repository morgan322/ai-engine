cmake_minimum_required(VERSION 3.10)
project(MobilenetSSDInference)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_FIND_ROOT_PATH /usr/local/x86_64-nvidia-gnu)
set(ENV{PKG_CONFIG_PATH} /usr/local/x86_64-nvidia-gnu/lib/pkgconfig:$ENV{PKG_CONFIG_PATH})
# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 如果没有找到 OpenCV，CMake 会报错
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 需要使用的源文件
set(SOURCE_FILES SSDModel.cpp)

# 创建可执行文件
add_executable(mobilenet_ssd_inference ${SOURCE_FILES})

# 包含 OpenCV 头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接 OpenCV 库
target_link_libraries(mobilenet_ssd_inference ${OpenCV_LIBS})
