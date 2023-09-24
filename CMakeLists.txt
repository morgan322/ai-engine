cmake_minimum_required(VERSION 3.5)

project(CmakeTestProject VERSION 0.1.0)

project(AI_H63) 
set(CMAKE_CXX_STANDARD 11)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)
aux_source_directory(./src dir_src) #指定路径，并产生变量${dir_src}
add_executable(AI_H63 ${dir_src}) #指定代码目录

# # set(CMAKE_CXX_STANDARD 11)
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# # # 指定编译器
# # set(CMAKE_CXX_COMPILER "g++")

# # # 指定 C++ 标准
# # set(CMAKE_CXX_STANDARD 11)

# # # 指定要编译的源文件和头文件
# # file(GLOB_RECURSE SOURCE_FILES 
# #     ${CMAKE_SOURCE_DIR}/src/*.c
# # )

# # )

# # # 指定生成位置
# # set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)

# # # 指定生成可执行文件
# # add_executable(executable_name ${SOURCE_FILES})



# set(SOURCES
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/Logger.h
# )

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add_executable(main ${SOURCES})