cmake_minimum_required(VERSION 3.18)
project(yolofast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_BUILD_TYPE Release)

#使用cmake文件夹下的FindTensorRT.cmake（默认TensorRT在/usr/下，比如Jetson）
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#如果在/usr下找不到，则手动设置TensorRT_DIR，Jetson可以不管这里
if (NOT TensorRT_DIR)
    set(TensorRT_DIR /home/ad/TensorRT-10.7.0.23/)
endif()

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(yolofast
    src/main.cc
    src/trt_engine.cc
    src/yolo.cc
    src/yolov8.cc
    src/yolov10.cc
    src/int8_calibrator.cc
    src/yolov5a.cc
)

target_link_libraries(yolofast PUBLIC ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(yolofast PUBLIC include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${${OpenCV_INCLUDE_DIRS}})