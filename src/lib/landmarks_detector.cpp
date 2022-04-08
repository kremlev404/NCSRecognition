/*
 * Performed by Anton Kremlev
 */

#include "landmarks_detector.hpp"

std::vector<float> LandmarkDetector::detect(const cv::Mat &image) {
    static auto net = getNet("LandmarkDetector");

    std::vector<float> detected_landmarks;
    cv::Mat resized_frame;

    cv::resize(image, resized_frame, net_size);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, net_size, mean, swapRB);
    net->setInput(inputBlob);
    cv::Mat outBlob = net->forward();

    // landmark regression. Output is 10 coordinates x0, y0, x1, y1, ..., x4, y4
    for (int i = 0; i < 10; i++) {
        detected_landmarks.push_back(outBlob.reshape(1, 1).at<float>(0, i));
    }

    return detected_landmarks;
}

