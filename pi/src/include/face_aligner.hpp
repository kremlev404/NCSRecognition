#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


class FaceAligner {
private:
    static std::vector<float> landmarks_ref;

    std::vector<cv::Point2f> coordsToPoints(std::vector<float> &pointsCoords);

public:
    cv::Mat align(cv::Mat image, std::vector<float> &landmarks);
};

