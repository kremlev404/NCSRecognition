#pragma once
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ie.hpp"

class FaceDetector : public IE {
private:
    cv::Size2i netSize;
    float confidence_threshold;
    double scale;
    cv::Scalar_<double> &&mean;
    bool swapRB;
public:
    FaceDetector(const std::string& xml,
                 const std::string& bin,
                 const std::string& device,
                 float confidence_threshold = 0.5,
                 int inputWidth = 300,
                 int inputHeight = 300,
                 double scale = 1.0,
                 cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
                 bool swapRB = false);

	std::vector<cv::Rect> detect(cv::Mat image);

};