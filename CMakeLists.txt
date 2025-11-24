cmake_minimum_required(VERSION 3.18)
project(yolofast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_BUILD_TYPE Release)

# Add module in charge of finding TensorRT
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

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