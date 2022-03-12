/*#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ie.hpp"

class FaceDetector : private IE {

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

	std::vector<cv::Rect> detect(const cv::Mat& image);

};*/
#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class FaceDetector{
private:
    cv::String modelPath;
    cv::String configPath;
    cv::Size netSize;
    cv::Scalar mean;
    bool swapRB;
    double scale;
    float confidence_threshold;
    cv::dnn::Net net;
public:
    FaceDetector(cv::String modelPath, cv::String configPath,
                 float confidence_threshold = 0.7,
                 int inputWidth = 384,
                 int inputHeight = 672 ,
                 double scale = 1.0,
                 cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
                 bool swapRB= false,
                 int backEnd = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE,
                 int target = cv::dnn::DNN_TARGET_MYRIAD);
    std::vector<cv::Rect> detect(const cv::Mat& image);
};