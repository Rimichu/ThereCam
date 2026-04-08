cmake_minimum_required(VERSION 3.16)
project(ThereCam)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT_DIR "$ENV{HOME}/libs/onnxruntime")
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT_DIR}/include
)

# rtmidi
set(RTMIDI_ROOT_DIR "/opt/homebrew/opt/rtmidi")
include_directories(${RTMIDI_ROOT_DIR}/include)

add_executable(ThereCam
    src/main.cpp
    src/utilities/utils.cpp
    src/handTracking/inference.cpp
)

target_compile_definitions(ThereCam PRIVATE
    THERECAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/AIModel/epoch60.onnx"
)

target_link_libraries(ThereCam
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.dylib
    ${RTMIDI_ROOT_DIR}/lib/librtmidi.dylib
)