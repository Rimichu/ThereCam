#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include "types.h"
#include "constants.h"

int initCamera(cv::VideoCapture& cap);
Hand getScaledHand(const Hand& hand, const int frameWidth, const int frameHeight);
std::vector<std::vector<float>> getSubmatrix(const std::vector<std::vector<float>>& matrix, size_t exclude_row, size_t exclude_col);
float determinant(const std::vector<std::vector<float>>& matrix);

#endif // UTILS_H