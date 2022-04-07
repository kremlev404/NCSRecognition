#include "landmarks_detector.hpp"

#include <utility>

LandmarkDetector::LandmarkDetector(cv::String _modelPath, cv::String _configPath,
                                   int inputWidth,
                                   int inputHeight,
                                   double scale,
                                   cv::Scalar mean,
                                   bool swapRB,
                                   int backEnd,
                                   int target)
        : modelPath(std::move(_modelPath)),
          configPath(std::move(_configPath)),
          netSize(cv::Size(inputWidth, inputHeight)),
          scale(scale), mean(std::move(mean)), swapRB(swapRB),
          backEnd(backEnd),
          target(target) {
}

cv::dnn::Net LandmarkDetector::getNet() {
    static auto net = cv::dnn::Net::readFromModelOptimizer(modelPath, configPath);
    static auto is_setted = false;
    if (!is_setted) {
        net.setPreferableBackend(backEnd);
        net.setPreferableTarget(target);
        std::vector available_backends = cv::dnn::getAvailableBackends();
        auto it = available_backends.begin();
        bool target_founded = false;
        while (it != available_backends.end()) {
            auto tg = cv::dnn::getAvailableTargets(it->first);
            if (std::find(tg.begin(), tg.end(), cv::dnn::Target::DNN_TARGET_MYRIAD) != tg.end()) {
                target_founded = true;
                break;
            }
            it++;
        }
        if (!target_founded) {
            throw std::invalid_argument("LandmarkDetector didn't found target");
        } else {
            std::cout << "LandmarkDetector created\n";
        }
        is_setted = true;
    }
    return net;
}

std::vector<float> LandmarkDetector::detect(const cv::Mat &image) {
    std::vector<float> detected_landmarks;
    auto net = getNet();
    cv::Mat resized_frame;
    cv::resize(image, resized_frame, netSize);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);
    net.setInput(inputBlob);

    cv::Mat outBlob = net.forward();
    for (int i = 0; i < 10; i++) { // landmark regression. Output is 10 coordinates x0, y0, x1, y1, ..., x4, y4
        detected_landmarks.push_back(outBlob.reshape(1, 1).at<float>(0, i));
    }

    return detected_landmarks;
}

