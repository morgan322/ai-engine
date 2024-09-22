cmake_minimum_required(VERSION 3.10)
project(AI_EXE)

# 递归获取当前目录及子目录下所有头文件
file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/**/*.hpp")

# 设置包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 添加可执行文件或库
add_executable(AI_EXE main_test.cpp ${HEADER_FILES}) 
