/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <opencv2/core/mat.hpp>

enum DetectorType {
    face_detection_retail_0001,
    face_detection_retail_0004,
    haar_cascade
};

class IDetector {
public:
    virtual std::vector<cv::Rect> detect(const cv::Mat &image) = 0;
};

std::shared_ptr<IDetector> build_detector(const DetectorType &type, const cv::String &xml, const cv::String &bin);