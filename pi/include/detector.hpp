#pragma once

#include "face_detectorDNN.hpp"

enum DetectorType {
    face_detection_retail_0001,
    face_detection_retail_0004,
    haar_cascade
};

// Detector factory function
API std::shared_ptr<FaceDetectorDNN>
build_detector(DetectorType type, const cv::String &xml, const cv::String &bin);