#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>

#include "macros.h"

enum ClassifierType {
    face_reidentification_retail_0095
};

typedef std::vector<float> FaceDescriptor;

// Interface of a classification
class Classifier {
public:
    virtual float compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person,
                                    float confidence_threshold = 0.34,
                                    float votes_threshold = 0.5) = 0;

    virtual FaceDescriptor embed(const cv::Mat &face) = 0;

    virtual ~Classifier() = default;
};

// Classification factory function
DEFAULT_VISIBILITY
std::shared_ptr<Classifier> build_classifier(ClassifierType type, const std::string &xml, const std::string &bin, const std::string &device);
