/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

#include "iclassifier.hpp"
#include "vino_net.hpp"

class FaceRecognizer : public IClassifier, protected VinoNet {
private:
    static float cosSimilarity(const std::vector<float> &first, const std::vector<float> &second);

public:
    FaceRecognizer(cv::String model_path, cv::String config_path) : VinoNet(std::move(model_path),
                                                                            std::move(config_path),
                                                                            128,
                                                                            128,
                                                                            1.0) {}

    std::vector<float> embed(const cv::Mat &image) override;

    float compareDescriptors(const std::vector<float> &initial_person, const std::vector<float> &compare_person) override;
};