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
#include <utility>
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

    const cv::Scalar color = cv::Scalar(211, 235, 0);
    const cv::Scalar known_color = cv::Scalar(211, 235, 0);
    const cv::Scalar unknown_color = cv::Scalar(86, 5, 247);

    float avg_fps = 0;
    const cv::Size window_size = cv::Size(480, 720);
    bool use_gray_filter;

    std::vector<float> getEmbed(const std::string &path_to_image);

    void reset();

public:
    float getAvgFps() const;

    void initBD(const std::string &string);

    void play(const bool &gui,const bool &flip, const std::shared_ptr<cv::VideoCapture> &capture);

    CoreExecutor(std::shared_ptr<IClassifier> classifier,
                 std::shared_ptr<IDetector> face_detector,
                 std::shared_ptr<FaceAligner> aligner,
                 std::shared_ptr<LandmarkDetector> landmark_detector,
                 std::shared_ptr<IGPIO> gpio_controller,
                 const int &update_period,
                 const bool &to_gray_filter);
};

struct Stats {
    int frame;
    std::string id;
    double prob;
    float fps;

    Stats(int frame,
          std::string id,
          double prob,
          float fps
    ) : frame(frame), id(std::move(id)), prob(prob), fps(fps) {}
};