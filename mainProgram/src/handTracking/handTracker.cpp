#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct Keypoint {
    float x;
    float y;
    float confidence;
};

class HandTracker {
    Ort::Session session;
    Ort::Env env;
    public:
        HandTracker(const std::string& model_path);

        std::vector<Keypoint> detect(const cv::Mat& frame) {
            // 1. Preprocess: resize, normalize, NCHW format
            
            // 2. Run session.Run(...)
            // 3. Parse output tensor → 21 keypoints
        }
};