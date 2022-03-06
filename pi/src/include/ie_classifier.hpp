#pragma once

#include "classifier.hpp"
#include "ie.hpp"

class IEClassifier: public Classifier, public IE {
public:
    IEClassifier(const std::string& xml, const std::string& bin, const std::string& device);
    float distance(const FaceDescriptor& desc1, const FaceDescriptor& desc2) override;
    FaceDescriptor embed(const cv::Mat& face) override;
    ~IEClassifier() override;
};
