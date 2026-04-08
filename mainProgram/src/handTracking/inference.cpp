#include "inference.h"
#include "../utilities/constants.h"
#include "../utilities/utils.h"

/**
 * Constructor for HandInference class
 * Initializes ONNX Runtime environment and session with the provided model path
 * @param model_path - Path to the ONNX model file
 */
HandInference::HandInference(const std::string& model_path) : 
    env(Ort::Env{ORT_LOGGING_LEVEL_WARNING, "HandInference"}), 
    session(env, model_path.c_str(), Ort::SessionOptions{}) {    
}

/**
 * Runs inference on the provided frame
 * @param frame - The input frame
 * @return A vector of detected hands
 */
std::vector<Hand> HandInference::runInference(cv::Mat& frame) {
    // 1. Preprocess: resize, normalize, NCHW format
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        SCALE_FACTOR,                   // Scale factor
        cv::Size(IMG_SIZE, IMG_SIZE),   // Size expected by the model
        cv::Scalar(0, 0, 0),            // Mean subtraction (if needed)
        true,                           // Swap BGR to RGB
        false                           // Don't crop
    );

    // Create Tensor (NCHW format)
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        reinterpret_cast<float*>(blob.data),
        input_tensor_size,
        MODEL_INPUT_DIMENSIONS.data(),
        MODEL_INPUT_DIMENSIONS.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = {session.GetInputNameAllocated(0, allocator)};
    Ort::AllocatedStringPtr output_name_ptr = {session.GetOutputNameAllocated(0, allocator)};
    std::vector<const char*> input_names = {input_name_ptr.get()};
    std::vector<const char*> output_names = {output_name_ptr.get()};

    std::vector<Ort::Value> output_tensor = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );

    return postProcessOutput(output_tensor[0]);
}

/**
 * Post-processes the output tensor to extract hand detections and keypoints
 * @param tensor - The output tensor from the model
 * @return A vector of detected hands with their keypoints and attributes
 */
std::vector<Hand> HandInference::postProcessOutput(Ort::Value& tensor) {
    float* out_data = tensor.GetTensorMutableData<float>();
    
    std::vector<Hand> hands;
    for (int i = 0; i < TENSOR_DETECTIONS; i++) {
        float confidence = out_data[i * TENSOR_VALUES_PER_DETECTION + 4];
        if (confidence > CONFIDENCE_THRESHOLD) {
            Hand hand = getHandFromTensor(out_data, i);
            hands.push_back(hand);
        }
        if (hands.size() >= MAX_HANDS) {
            break; // Limit to max hands
        }
    }


    return hands;
}

/**
 * Determines the side of the hand (LEFT, RIGHT, UNKNOWN) based on keypoint positions
 * @param hand - The hand object to update with the determined side
 */
void HandInference::determineHandSide(Hand& hand) {
    // Check if keypoints are available
    if (hand.keypoints.size() < KEYPOINTS_PER_HAND) {
        hand.side = HandSide::UNKNOWN;
        return;
    }

    Keypoint wrist = hand.keypoints[0];
    Keypoint index_mcp = hand.keypoints[5];
    Keypoint pinky_mcp = hand.keypoints[17];

    if (wrist.confidence        < CONFIDENCE_THRESHOLD || 
        index_mcp.confidence    < CONFIDENCE_THRESHOLD || 
        pinky_mcp.confidence    < CONFIDENCE_THRESHOLD) {
        return; // Not enough confidence to determine hand side so leave as UNKNOWN
    }
    
    std::vector<std::vector<float>> matrix = {
        {wrist.x, wrist.y, 1},
        {index_mcp.x, index_mcp.y, 1},
        {pinky_mcp.x, pinky_mcp.y, 1}
    };

    float det = determinant(matrix);
    if (det > 0) {
        hand.side = HandSide::LEFT;
    } else if (det < 0) {
        hand.side = HandSide::RIGHT;
    }
}

/**
 * Extracts hand attributes and keypoints from the output tensor for a specific hand index
 * @param tensor - The output tensor from the model
 * @param hand_index - The index of the hand detection to extract
 * @return A Hand object populated with the bounding box, confidence, and keypoints
 */
Hand HandInference::getHandFromTensor(float* tensor, int hand_index) {
    Hand hand;
    int base = hand_index * TENSOR_VALUES_PER_DETECTION;

    hand.x = tensor[base + 0];
    hand.y = tensor[base + 1];
    hand.width = tensor[base + 2];
    hand.height = tensor[base + 3];
    hand.confidence = tensor[base + 4];

    // Extract keypoints for this hand
    for (int k = 0; k < KEYPOINTS_PER_HAND; k++) {
        hand.keypoints.push_back(getKeypointFromTensor(tensor, hand_index, k));
    }

    HandInference::determineHandSide(hand);

    return hand;
}

/**
 * Extracts a specific keypoint's attributes from the output tensor for a given hand and keypoint index
 * @param tensor - The output tensor from the model
 * @param hand_index - The index of the hand detection
 * @param keypoint_index - The index of the keypoint to extract (0-20)
 * @return A Keypoint object populated with x, y coordinates and confidence
 */
Keypoint HandInference::getKeypointFromTensor(float* tensor, int hand_index, int keypoint_index) {
    Keypoint kp;
    int base = hand_index * TENSOR_VALUES_PER_DETECTION + TENSOR_KEYPOINT_OFFSET + keypoint_index * 3;

    kp.x = tensor[base + 0];
    kp.y = tensor[base + 1];
    kp.confidence = tensor[base + 2];
    return kp;
}