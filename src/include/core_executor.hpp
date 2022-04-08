/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <map>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "classifier.hpp"
#include "detector.hpp"
#include "face_aligner.hpp"
#include "landmarks_detector.hpp"
#include "data/face_data.hpp"

class CoreExecutor {
private:
    std::shared_ptr<Classifier> classifier;
    std::shared_ptr<Detector> face_detector;
    std::shared_ptr<FaceAligner> aligner;
    std::shared_ptr<LandmarkDetector> landmark_detector;
    std::map<std::string, std::vector<float>> people;
    //
    // kremlev ->  ( kremlev1.png:descriptor ) ..
    // mmusk -> ( musk1.png:descriptor ) ..
    //
    std::map<std::string, std::vector<FaceData>> peoples;
    cv::Scalar color = cv::Scalar(211, 235, 0);
    cv::Scalar known_color = cv::Scalar(211, 235, 0);
    cv::Scalar unknown_color = cv::Scalar(86, 5, 247);
    float avg_fps = 0;

    std::vector<float> getEmbed(const std::string& path_to_image);
public:
    float getAvgFps() const;

    void initBD(const std::string &string);

    void play(bool gui, bool flip, cv::VideoCapture capture);

    CoreExecutor(std::shared_ptr<Classifier> classifier,
                 std::shared_ptr<Detector> face_detector,
                 std::shared_ptr<FaceAligner> aligner,
                 std::shared_ptr<LandmarkDetector> landmark_detector);

};