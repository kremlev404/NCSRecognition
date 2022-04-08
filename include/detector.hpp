/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <opencv2/core/mat.hpp>

#include "macros.hpp"

enum DetectorType {
    face_detection_retail_0001,
    face_detection_retail_0004,
    haar_cascade
};

// Interface of a detector
class Detector {
public:
    virtual std::vector<cv::Rect> detect(const cv::Mat& image) = 0;
};

// Detector factory function
DEFAULT_VISIBILITY
std::shared_ptr<Detector> build_detector(DetectorType type, const cv::String &xml, const cv::String &bin);