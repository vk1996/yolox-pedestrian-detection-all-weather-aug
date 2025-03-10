cmake_minimum_required(VERSION 3.10)
project(yolox_onnx)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set ONNX Runtime paths (update these for your system)
# download from https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
set(ONNXRUNTIME_DIR "onnxruntime-linux-x64-1.20.1")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Create executable
add_executable(yolox_onnx yolox_onnx.cpp)

# Link OpenCV and ONNX Runtime
target_link_libraries(yolox_onnx ${OpenCV_LIBS} onnxruntime)
