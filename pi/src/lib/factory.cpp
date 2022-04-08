#include "face_recognizer.hpp"
#include "face_detector.hpp"
#include "face_detector_cascade.hpp"

std::shared_ptr<Classifier>
build_classifier(ClassifierType type, const std::string &xml, const std::string &bin, const std::string &device) {
    if (type == ClassifierType::face_reidentification_retail_0095) {
        return std::make_shared<FaceRecognizer>(xml, bin);
    } else {
        throw std::runtime_error("Unknown classifier type");
    }
}

std::shared_ptr<Detector> build_detector(DetectorType type, const cv::String &xml, const cv::String &bin) {
    if (type == DetectorType::face_detection_retail_0001) {
        return std::make_shared<FaceDetector>(xml, bin, 0.7, 384, 672);
    } else if (type == DetectorType::face_detection_retail_0004) {
        return std::make_shared<FaceDetector>(xml, bin, 0.5, 300, 300);
    } else if (type == DetectorType::haar_cascade) {
        return std::make_shared<FaceDetectorCascade>(xml);
    } else {
        throw std::runtime_error("Unknown detector type");
    }
}
