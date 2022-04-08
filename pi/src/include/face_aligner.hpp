#pragma once

#include <vector>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class FaceAligner {
private:
    static std::vector<float> landmarks_ref;

    std::vector<cv::Point2f> coordsToPoints(std::vector<float> &points_coords);

public:
    cv::Mat align(const cv::Mat& image, std::vector<float> &landmarks);
};

