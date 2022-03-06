#include "ie_classifier.hpp"

//Factory method
std::shared_ptr<Classifier> build_classifier(ClassifierType type, const std::string& xml, const std::string& bin, const std::string& device) {
    if (type == ClassifierType::IE_Facenet_V1) {
        return std::make_shared<IEClassifier>(xml, bin, device);
    } else {
        throw std::runtime_error("Unknown classifier type");
    }
}