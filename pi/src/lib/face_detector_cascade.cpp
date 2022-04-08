#include "face_detector_cascade.hpp"

#include <utility>

FaceDetectorCascade::FaceDetectorCascade(cv::String model_path,
                                         double scale_factor,
                                         int min_neighbors,
                                         int flags,
                                         cv::Size min_size)
        : model_path(std::move(model_path)),
          scale_factor(scale_factor),
          min_neighbors(min_neighbors),
          flags(flags),
          min_size(std::move(min_size)) {
}

cv::CascadeClassifier FaceDetectorCascade::getNet() {
    static cv::CascadeClassifier cascade;
    static auto is_setted = false;
    if (!is_setted) {
        cascade.load(model_path);
        is_setted = true;
    }
    return cascade;
}

std::vector<cv::Rect> FaceDetectorCascade::detect(const cv::Mat &image) {
    auto cascade = getNet();
    std::vector<cv::Rect> output;
    cascade.detectMultiScale(image, output, scale_factor, min_neighbors, flags, min_size);
    return output;
}
