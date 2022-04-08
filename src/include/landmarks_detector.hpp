/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include "vino_net.hpp"

class LandmarkDetector : protected VinoNet {
public:
    LandmarkDetector(cv::String model_path, cv::String config_path,
                     int input_width = 48,
                     int input_height = 48
    ) : VinoNet(std::move(model_path), std::move(config_path), input_width, input_height, 1) {}

    std::vector<float> detect(const cv::Mat &image);
};