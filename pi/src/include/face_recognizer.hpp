#pragma once

#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "classifier.hpp"

class FaceRecognizer : public Classifier{
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
    static float cosSimilarity(std::vector<float> &first, std::vector<float> &second);
public:
    FaceRecognizer(cv::String modelPath, cv::String configPath,
                   int inputWidth = 128,
                   int inputHeight = 128,
                   double scale = 1.0,
                   cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
                   bool swapRB = false,
                   int backEnd = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE,
                   int target = cv::dnn::DNN_TARGET_MYRIAD);

    std::vector<float> embed(const cv::Mat &image) override;

    float compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person,
                                 float confidence_threshold ,
                                 float votes_threshold ) override; // true - it's the same person
};