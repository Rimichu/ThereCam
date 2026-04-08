#include "utils.h"

/**
 * Initializes the camera and checks if it is opened successfully.
 * @param cap Reference to the VideoCapture object to be initialized.
 * @return 0 if successful, -1 if there was an error opening the camera.
 */
int initCamera(cv::VideoCapture& cap) {
    cap.open(DEFAULT_CAMERA_INDEX); // Ensure the camera is opened
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    std::cout << "Camera opened successfully" << std::endl;
    return 0;
}

/**
 * Scales the detected hand's bounding box and keypoints from the model's input size back to the original frame size, accounting for letterboxing.
 * @param hand The detected hand with coordinates relative to the model's input size.
 * @param frameWidth The width of the original camera frame.
 * @param frameHeight The height of the original camera frame.
 * @return A Hand object with coordinates scaled to the original frame size.
 */
Hand getScaledHand(const Hand& hand, const int frameWidth, const int frameHeight) {
    Hand scaledHand = hand;
    
    float xScale = (float)frameWidth / (float)IMG_SIZE;
    float yScale = (float)frameHeight / (float)IMG_SIZE;

    scaledHand.x = (hand.x) * xScale;
    scaledHand.y = (hand.y) * yScale;
    scaledHand.width = (hand.width - hand.x) * xScale;
    scaledHand.height = (hand.height - hand.y) * yScale;
    
    for (Keypoint& kp : scaledHand.keypoints) {
        kp.x = (kp.x) * xScale;
        kp.y = (kp.y) * yScale;
    }
    
    return scaledHand;
}

/**
 * Generates a submatrix by excluding a specified row and column from the original matrix, used for determinant calculation.
 * @param matrix The original square matrix.
 * @param exclude_row The index of the row to exclude.
 * @param exclude_col The index of the column to exclude.
 * @return A new matrix with the specified row and column removed.
 */
std::vector<std::vector<float>> getSubmatrix(const std::vector<std::vector<float>>& matrix, size_t exclude_row, size_t exclude_col) {
    std::vector<std::vector<float>> submatrix;
    for (size_t i = 0; i < matrix.size(); i++) {
        if (i == exclude_row) continue;
        std::vector<float> row;
        for (size_t j = 0; j < matrix[i].size(); j++) {
            if (j == exclude_col) continue;
            row.push_back(matrix[i][j]);
        }
        submatrix.push_back(row);
    }
    return submatrix;
}

/**
 * Calculates the determinant of a square matrix using recursive cofactor expansion.
 * @param matrix The square matrix for which to calculate the determinant.
 * @return The determinant value of the matrix.
 * @throws std::invalid_argument if the matrix is not square.
 */
float determinant(const std::vector<std::vector<float>>& matrix) {
    if (matrix.size() != matrix[0].size()) {
        throw std::invalid_argument("Matrix must be square");
    }
    
    if (matrix.size() == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    float det = 0;
    bool sign = true; // To track the sign for cofactor expansion
    for (size_t i = 0; i < matrix.size(); i++) {
        det += (sign ? 1 : -1) * matrix[0][i] * determinant(getSubmatrix(matrix, 0, i));
        sign = !sign;
    }
    return det;
}