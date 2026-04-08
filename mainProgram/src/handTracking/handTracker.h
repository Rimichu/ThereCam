#ifndef HAND_TRACKER_H
#define HAND_TRACKER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Keypoint;

class HandTracker {
    public:
        HandTracker(const std::string& model_path);
        std::vector<Keypoint> detect(const cv::Mat& frame);

};

#endif // HAND_TRACKER_H