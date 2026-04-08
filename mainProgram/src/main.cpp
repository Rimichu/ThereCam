#include <onnxruntime_cxx_api.h>
#include "utilities/utils.h"
#include "utilities/constants.h"
#include "handTracking/inference.h"

#ifndef THERECAM_MODEL_PATH
#define THERECAM_MODEL_PATH "src/AIModel/epoch60.onnx"
#endif

int main() {
    cv::Mat frame;
    cv::VideoCapture cap;

    if (initCamera(cap) != 0) {
        return CAMERA_INIT_FAILURE;
    }

    std::cout << "Using model at: " << THERECAM_MODEL_PATH << std::endl;
    HandInference handInference(THERECAM_MODEL_PATH);

    while (true) {
        // Read frame from Camera
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame" << std::endl;
            break;
        }

        // Run Inference
        std::vector<Hand> hands = handInference.runInference(frame);

        // Get scaled hand coordinates and draw results
        for (Hand& hand : hands) {
            hand = getScaledHand(hand, frame.cols, frame.rows);
            cv::rectangle(frame, cv::Rect(hand.x, hand.y, hand.width, hand.height), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, hand.side.toString() + " Confidence: " + std::to_string(hand.confidence), cv::Point(hand.x, hand.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
            for (const Keypoint& kp : hand.keypoints) {
                cv::circle(frame, cv::Point(kp.x, kp.y), 5, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // Display frame
        cv::imshow("Camera Feed", frame);
        
        // Exit on key press
        if (cv::waitKey(FRAME_DELAY_MS) >= 0) {
            break; // Exit on any key press
        }
    }

    return SUCCESS;
}