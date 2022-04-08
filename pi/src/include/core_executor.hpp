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

class CoreExecutor {
private:
    std::shared_ptr<Classifier> classifier;
    std::shared_ptr<Detector> face_detector;
    std::shared_ptr<FaceAligner> aligner;
    std::shared_ptr<LandmarkDetector> landmark_detector;
    std::map<std::string, std::vector<float>> people;
public:
    void initBD(const std::string &string);

    void play(bool gui, bool flip, cv::VideoCapture capture);

    CoreExecutor(std::shared_ptr<Classifier> classifier,
                 std::shared_ptr<Detector> face_detector,
                 std::shared_ptr<FaceAligner> aligner,
                 std::shared_ptr<LandmarkDetector> landmark_detector);

};