#include "ie_classifier.hpp"
#include "face_recognizer.hpp"
#include "detector.hpp"

//Factory method
std::shared_ptr<Classifier>
build_classifier(ClassifierType type, const std::string &xml, const std::string &bin, const std::string &device) {
    if (type == ClassifierType::IE_Facenet_V1) {
        return std::make_shared<IEClassifier>(xml, bin, device);
    } else if (type == ClassifierType::face_reidentification_retail_0095) {
        return std::make_shared<FaceRecognizer>(xml, bin);
    } else {
        throw std::runtime_error("Unknown classifier type");
    }
}

std::shared_ptr<FaceDetectorDNN> build_detector(DetectorType type, const cv::String &xml, const cv::String &bin) {
    if (type == DetectorType::face_detection_retail_0001) {
        return std::make_shared<FaceDetectorDNN>(xml, bin, 0.7, 384, 672);
    } else if (type == DetectorType::face_detection_retail_0004) {
        return std::make_shared<FaceDetectorDNN>(xml, bin, 0.5, 300, 300);
    } else {
        throw std::runtime_error("Unknown detector type");
    }
}
