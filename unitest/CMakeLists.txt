cmake_minimum_required(VERSION 3.5)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ 3rdparty
# set(3RDPARTY_LIBS "../")

# ---[ opencv
set(OpenCV_FIND_QUIETLY true)
# find_package(OpenCV REQUIRED) //this will include opencv_ts which result in crash on centos
find_package(OpenCV OPTIONAL_COMPONENTS imgcodecs)
find_package(OpenCV OPTIONAL_COMPONENTS videoio)
set(optional_opencv_libs ${OpenCV_LIBS})
find_package(OpenCV REQUIRED core imgproc highgui features2d)
if(OpenCV_FOUND)
  if(imgcodecs_FOUND)
    list(APPEND OpenCV_LIBS optional_opencv_libs)
  endif()
  include_directories(${OpenCV_INCLUDE_DIRS})
  message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "opencv libraries: ${OpenCV_LIBS}")
else()
  message(FATAL_ERROR "opencv not found!")
endif()
list(APPEND 3RDPARTY_LIBS ${OpenCV_LIBS})

# ---[ gflags
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGFlags.cmake)
include_directories(${GFLAGS_INCLUDE_DIRS})
list(APPEND 3RDPARTY_LIBS ${GFLAGS_LIBRARIES})

# ---[ glog
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGlog.cmake)
include_directories(${GLOG_INCLUDE_DIRS})
list(APPEND 3RDPARTY_LIBS ${GLOG_LIBRARIES})

set(AI_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/)

# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../samples/cmake)
# ---[ FFmpeg
find_package(FFmpeg REQUIRED)
if (FFMPEG_FOUND)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  list(APPEND 3RDPARTY_LIBS ${FFMPEG_LIBRARIES})
  set(HAVE_FFMPEG true)
  add_compile_definitions(HAVE_FFMPEG) 
  if (WITH_FFMPEG_AVDEVICE)
      add_compile_definitions(HAVE_FFMPEG_AVDEVICE)
  endif ()
else ()
  message(FATAL_ERROR "FFmpeg not found!")
endif ()

find_package(JPEG REQUIRED)


file(GLOB test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

message(STATUS "@@@@@@@@@@@ Target : tests_ai")
add_executable(tests_ai ${test_srcs})
add_sanitizers(tests_ai)

target_include_directories(tests_ai PRIVATE
                           ${FFMPEG_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/src/codec
                           ${EASYDK_ROOT_DIR}/src/platform)

target_link_libraries(tests_ai PRIVATE gtest gtest_main ${3RDPARTY_LIBS} ${JPEG_LIBRARIES} pthread)
target_compile_options(tests_ai PRIVATE "-Wno-deprecated-declarations")
target_compile_definitions(tests_ai PRIVATE ${EDK_DEFINITIONS})

install(TARGETS tests_ai RUNTIME DESTINATION bin)
