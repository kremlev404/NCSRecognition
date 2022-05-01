/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>

enum ClassifierType {
    face_reindefication_retail_0095
};

// Interface of a classification
class IClassifier {
public:
    virtual float compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person) = 0;

    virtual std::vector<float> embed(const cv::Mat &face) = 0;

    virtual ~IClassifier() = default;
};

// Classification factory function
std::shared_ptr<IClassifier> build_classifier(ClassifierType type, const std::string &xml, const std::string &bin, const std::string &device);
