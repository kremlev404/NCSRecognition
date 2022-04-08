/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "detector.hpp"

class FaceDetectorCascade : public Detector {
private:
    double scale_factor;
    int min_neighbors;
    int flags;
    cv::Size min_size;

    std::shared_ptr<cv::CascadeClassifier> cascade;

public:
    explicit FaceDetectorCascade(const cv::String &model_path,
                                 double scale_factor = 1.1,
                                 int min_neighbors = 3,
                                 int flags = 0,
                                 cv::Size min_size = cv::Size(150, 150));

    std::vector<cv::Rect> detect(const cv::Mat &image) override;
};