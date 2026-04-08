#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <array>

// Exit Codes
constexpr int SUCCESS = 0;
constexpr int CAMERA_INIT_FAILURE = -1;

// Camera
constexpr int DEFAULT_CAMERA_INDEX = 0;
constexpr int FRAME_DELAY_MS = 5;
constexpr int CAMERA_WIDTH = 1920;
constexpr int CAMERA_HEIGHT = 1080;

// Inference
constexpr int IMG_SIZE = 640; // Assumes square input
constexpr std::array<int64_t, 4> MODEL_INPUT_DIMENSIONS = {1, 3, IMG_SIZE, IMG_SIZE}; // Batch size, Channels, Height, Width
constexpr int input_tensor_size = MODEL_INPUT_DIMENSIONS[0] * MODEL_INPUT_DIMENSIONS[1] * MODEL_INPUT_DIMENSIONS[2] * MODEL_INPUT_DIMENSIONS[3];
constexpr double SCALE_FACTOR = 1.0 / 255.0;
constexpr float CONFIDENCE_THRESHOLD = 0.5;
constexpr float IOU_THRESHOLD = 0.3;
constexpr int MAX_HANDS = 2;
constexpr uint TENSOR_DETECTIONS = 300;
constexpr uint TENSOR_VALUES_PER_DETECTION = 69;
constexpr uint TENSOR_KEYPOINT_OFFSET = 6;
constexpr uint KEYPOINTS_PER_HAND = 21;

#endif // CONSTANTS_H