#include "face_detector_cascade.hpp"

#include <utility>

FaceDetectorCascade::FaceDetectorCascade(cv::String modelPath,
                                         double scaleFactor,
                                         int minNeighbors,
                                         int flags,
                                         cv::Size minSize)
        : modelPath(std::move(modelPath)),
          scaleFactor(scaleFactor),
          minNeighbors(minNeighbors),
          flags(flags),
          minSize(std::move(minSize)) {
}

cv::CascadeClassifier FaceDetectorCascade::getNet() {
    static cv::CascadeClassifier cascade;
    static auto is_setted = false;
    if(!is_setted) {
        cascade.load(modelPath);
        is_setted = true;
    }
    return cascade;
}

std::vector<cv::Rect> FaceDetectorCascade::detect(const cv::Mat &image) {
    auto cascade = getNet();
    std::vector<cv::Rect> output;
    cascade.detectMultiScale(image, output, scaleFactor, minNeighbors, flags, minSize);
    return output;
}
