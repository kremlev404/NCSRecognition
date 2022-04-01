#pragma once

#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class LandmarkDetector {
private:
    cv::String modelPath;
    cv::String configPath;
    cv::Size netSize;
    cv::Scalar mean;
    bool swapRB;
    double scale;
    //cv::dnn::Net net;
    int backEnd;
    int target;
    cv::dnn::Net getNet();

public:
    LandmarkDetector(cv::String modelPath, cv::String configPath,
                     int inputWidth = 48,
                     int inputHeight = 48,
                     double scale = 1.0,
                     cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
                     bool swapRB = false,
                     int backEnd = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE,
                     int target = cv::dnn::DNN_TARGET_MYRIAD);

    std::vector<float> detect(const cv::Mat &image);
};