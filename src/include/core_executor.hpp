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

#include "iclassifier.hpp"
#include "idetector.hpp"
#include "face_aligner.hpp"
#include "landmarks_detector.hpp"
#include "data/face_data.hpp"
#include "firebase_interactor.hpp"
#include "timer.hpp"
#include "igpio.hpp"

class CoreExecutor {
private:
    //
    // kremlev ->  ( kremlev1.png:descriptor ) ..
    // musk ->     ( musk1.png:descriptor    ) ..
    //
    std::map<std::string, std::vector<FaceData>> peoples;

    std::shared_ptr<IClassifier> classifier;
    std::shared_ptr<IDetector> face_detector;
    std::shared_ptr<FaceAligner> aligner;
    std::shared_ptr<LandmarkDetector> landmark_detector;
    std::shared_ptr<IGPIO> gpio_controller;
    std::unique_ptr<FirebaseInteractor> firebase_interactor;
    std::unique_ptr<Timer> timer;

    cv::Scalar color = cv::Scalar(211, 235, 0);
    cv::Scalar known_color = cv::Scalar(211, 235, 0);
    cv::Scalar unknown_color = cv::Scalar(86, 5, 247);

    float avg_fps = 0;

    std::vector<float> getEmbed(const std::string &path_to_image);

public:
    float getAvgFps() const;

    void initBD(const std::string &string);

    void play(bool gui, bool flip, const std::shared_ptr<cv::VideoCapture> &capture);

    CoreExecutor(std::shared_ptr<IClassifier> classifier,
                 std::shared_ptr<IDetector> face_detector,
                 std::shared_ptr<FaceAligner> aligner,
                 std::shared_ptr<LandmarkDetector> landmark_detector,
                 std::shared_ptr<IGPIO> gpio_controller,
                 int update_period);
};