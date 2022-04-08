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

#include "classifier.hpp"
#include "vino_net.hpp"

class FaceRecognizer : public Classifier, protected VinoNet {
private:
    static float cosSimilarity(std::vector<float> &first, std::vector<float> &second);

public:
    FaceRecognizer(cv::String model_path, cv::String config_path) : VinoNet(std::move(model_path),
                                                                            std::move(config_path),
                                                                            128,
                                                                            128,
                                                                            1.0) {}

    std::vector<float> embed(const cv::Mat &image) override;

    float compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person) override;
};