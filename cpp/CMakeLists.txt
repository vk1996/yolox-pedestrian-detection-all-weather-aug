cmake_minimum_required(VERSION 3.10)
project(yolox-onnx)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set ONNX Runtime paths (update these for your system)
set(ONNXRUNTIME_DIR "onnxruntime-linux-x64-1.20.1")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Create executable
add_executable(yolox-onnx yolox-onnx.cpp)

# Link OpenCV and ONNX Runtime
target_link_libraries(yolox-onnx ${OpenCV_LIBS} onnxruntime)
