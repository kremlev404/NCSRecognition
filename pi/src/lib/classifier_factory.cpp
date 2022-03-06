#include "ie_facenet_v1.hpp"

std::shared_ptr<Classifier> build_classifier(ClassifierType type, const std::string& xml, const std::string& bin, const std::string& device) {
    if (type == ClassifierType::IE_Facenet_V1) {
        return std::make_shared<IEFacenet_V1>(xml, bin, device);
    } else {
        throw std::runtime_error("Unknown classifier type");
    }
}