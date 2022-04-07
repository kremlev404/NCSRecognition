#include "face_detector.hpp"

#include <utility>

FaceDetector::FaceDetector(cv::String _modelPath, cv::String _configPath,
                           float confidence_threshold,
                           int inputWidth,
                           int inputHeight,
                           double scale,
                           cv::Scalar mean,
                           bool swapRB,
                           int backEnd,
                           int target)
        : modelPath(std::move(_modelPath)),
          backEnd(backEnd),
          target(target),
          configPath(std::move(_configPath)),
          netSize(cv::Size(inputWidth, inputHeight)),
          confidence_threshold(confidence_threshold),
          scale(scale),
          mean(std::move(mean)),
          swapRB(swapRB) {
}

cv::dnn::Net FaceDetector::getNet() {
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
            throw std::invalid_argument("FaceDetector didn't found target");
        } else {
            std::cout << "FaceDetector created\n";
        }
        is_setted = true;
    }
    return net;
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat &image) {
    //cv::dnn::resetMyriadDevice();
    auto net = getNet();

    std::vector<cv::Rect> detected_objects;

    cv::Mat resized_frame;
    cv::resize(image, resized_frame, netSize);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);

    net.setInput(inputBlob);

    cv::Mat outBlob = net.forward();
    cv::Mat detection_as_mat(outBlob.size[2], outBlob.size[3], CV_32F, outBlob.ptr<float>());

    for (int i = 0; i < detection_as_mat.rows; i++) {
        float cur_confidence = detection_as_mat.at<float>(i, 2);
        int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * image.cols);
        int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * image.rows);
        int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * image.cols);
        int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * image.rows);
        cv::Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

        if (cur_confidence < confidence_threshold)
            continue;

        if (cur_rect.empty())
            continue;

        cur_rect = cur_rect & cv::Rect(cv::Point(), image.size());
        detected_objects.push_back(cur_rect);
    }
    return detected_objects;
}