#ifndef INFERENCE_H
#define INFERENCE_H

#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include "../utilities/types.h"

/**
 * Class responsible for running inference on the hand detection model
 * and post-processing the output to extract hand information.
 */
class HandInference {
    Ort::Env env;
    Ort::Session session;
    public:
        HandInference(const std::string& model_path);
        std::vector<Hand> runInference(cv::Mat& frame);
    private:
        std::vector<Hand> postProcessOutput(Ort::Value& tensor);
        void determineHandSide(Hand& hand);
        Hand getHandFromTensor(float* tensor, int hand_index);
        Keypoint getKeypointFromTensor(float* tensor, int hand_index, int keypoint_index);
};

#endif // INFERENCE_H