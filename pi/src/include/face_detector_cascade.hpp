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
    cv::String modelPath;
    double scaleFactor;
    int minNeighbors;
    int flags;
    cv::Size minSize;

    cv::CascadeClassifier getNet();

public:
    explicit FaceDetectorCascade(cv::String modelPath,
                        double scaleFactor = 1.1,
                        int minNeighbors = 3,
                        int flags = 0,
                        cv::Size minSize = cv::Size(150, 150));

    std::vector<cv::Rect> detect(const cv::Mat &image) override;
};
