#pragma once

#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

#include "detector.hpp"
#include "vino_net.hpp"

class FaceDetector : public Detector, protected VinoNet {
private:
    float confidence_threshold{};

public:
    FaceDetector(cv::String model_path, cv::String config_path,
                 float confidence_threshold,
                 int input_width,
                 int input_height,
                 double scale = 1.0);

    std::vector<cv::Rect> detect(const cv::Mat &image) override;
};
