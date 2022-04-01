#pragma once

#include "classifier.hpp"
#include "ie.hpp"

class IEClassifier: public Classifier, private IE {
public:
    IEClassifier(const std::string& xml, const std::string& bin, const std::string& device);
    float compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person,
                            float confidence_threshold ,
                            float votes_threshold ) override;
    FaceDescriptor embed(const cv::Mat& face) override;
    ~IEClassifier() override;
};
