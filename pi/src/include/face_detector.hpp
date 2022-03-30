#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ie.hpp"

class FaceDetector //: private IE {
{
private:
    InferenceEngine::ExecutableNetwork _executable;
    InferenceEngine::CNNNetwork _network;
    InferenceEngine::InferRequest _infer_request;
    InferenceEngine::Blob::Ptr _input;
    InferenceEngine::Blob::Ptr _output;
    cv::Size2i netSize;
    float confidence_threshold;
    cv::Scalar_<double> &&mean;
    bool swapRB;
    double scale;
    int maxProposalCount;
    int objectSize;
    std::string outputName;
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

};