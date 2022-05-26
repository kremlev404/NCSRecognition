/*
 * Performed by Anton Kremlev
 */

#include "face_detector_cascade.hpp"

#include <utility>

FaceDetectorCascade::FaceDetectorCascade(const cv::String &model_path,
                                         const double &scale_factor,
                                         const int &min_neighbors,
                                         const int &flags,
                                         cv::Size min_size) :
        scale_factor(scale_factor),
        min_neighbors(min_neighbors),
        flags(flags),
        min_size(std::move(min_size)) {
    cascade = std::make_shared<cv::CascadeClassifier>();
    cascade->load(model_path);
    std::cout << "FaceDetectorCascade created\n";
}


std::vector<cv::Rect> FaceDetectorCascade::detect(const cv::Mat &image) {
    std::vector<cv::Rect> output;
    cascade->detectMultiScale(image, output, scale_factor, min_neighbors, flags, min_size);
    return output;
}
