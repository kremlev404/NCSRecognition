/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>

enum ClassifierType {
    face_reindefication_retail_0095
};

class IClassifier {
public:
    virtual float compareDescriptors(const std::vector<float> &initial_person, const std::vector<float> &compare_person) = 0;

    virtual std::vector<float> embed(const cv::Mat &face) = 0;

    virtual cv::Size getInputSize() const = 0;

    virtual ~IClassifier() = default;
};

std::shared_ptr<IClassifier> build_classifier(const ClassifierType &type, const std::string &xml, const std::string &bin, const std::string &device);
