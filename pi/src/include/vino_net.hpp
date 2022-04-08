/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <opencv2/dnn.hpp>

class VinoNet {
protected:
    cv::String model_path;
    cv::String config_path;
    cv::Size net_size;
    cv::Scalar mean;
    bool swapRB;
    double scale;
    int backend;
    int target;
    std::shared_ptr<cv::dnn::Net> getNet(const std::string &class_name);

    VinoNet(cv::String model_path, cv::String config_path,
            int input_width,
            int input_height,
            double scale,
            cv::Scalar mean = cv::Scalar(0, 0, 0, 0),
            bool swapRB = false,
            int backend = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE,
            int target = cv::dnn::DNN_TARGET_MYRIAD);

};