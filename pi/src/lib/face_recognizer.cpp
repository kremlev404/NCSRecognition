#include "face_recognizer.hpp"

#include <utility>

FaceRecognizer::FaceRecognizer(cv::String modelPath, cv::String configPath,
                               int inputWidth,
                               int inputHeight,
                               double scale,
                               cv::Scalar mean,
                               bool swapRB,
                               int backEnd,
                               int target)
        : modelPath(std::move(modelPath)),
          configPath(std::move(configPath)),
          netSize(cv::Size(inputWidth, inputHeight)),
          scale(scale), mean(std::move(mean)), swapRB(swapRB),
          backEnd(backEnd), target(target) {
}

cv::dnn::Net FaceRecognizer::getNet() {
    static auto net = cv::dnn::Net::readFromModelOptimizer(modelPath, configPath);
    static auto isSetted = false;
    if (!isSetted) {
        net.setPreferableBackend(backEnd);
        net.setPreferableTarget(target);
        // std::cout << "InferenceEngineBackendType: " << cv::dnn::getInferenceEngineBackendType() << " CPU TYPE: "
        //          << cv::dnn::getInferenceEngineCPUType() << std::endl;
        std::vector ve = cv::dnn::getAvailableBackends();
        auto[back, pref] = ve[0];
        auto it = ve.begin();
        while (it != ve.end()) {
            std::cout << "FaceRecognizer Available Back: " << it->first << "Target: " << it->second << " " << std::endl;
            auto tg = cv::dnn::getAvailableTargets(it->first);
            auto targetIt = tg.begin();
            while (targetIt != tg.end()) {
                std::cout << "FaceRecognizer targetIt (0) is CPU, (3) is NCS:" << *targetIt << std::endl;
                targetIt++;
            }
            it++;
        }
        isSetted = true;
    }
    return net;
}

std::vector<float> FaceRecognizer::embed(const cv::Mat &image) {
    auto net = getNet();
    std::vector<cv::Rect> detected_objects;

    cv::Mat resized_frame;
    cv::resize(image, resized_frame, netSize);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);
    net.setInput(inputBlob);

    cv::Mat outBlob = net.forward();
    return std::vector<float>(outBlob.reshape(1, 1));
}

/*cosSimilarity = A*B/(|A|*|B|) */
float FaceRecognizer::cosSimilarity(std::vector<float> &first, std::vector<float> &second) {
    if (first.size() != second.size()) {
        throw "Vectors must have the same size";
    }
    size_t vec_size = first.size();
    float AB_numerator = 0;
    float AB_denominator = 0;
    float squares_sum_A = 0;
    float squares_sum_B = 0;

    for (size_t i = 0; i < vec_size; i++) {
        AB_numerator += first[i] * second[i];
        squares_sum_A += first[i] * first[i];
        squares_sum_B += second[i] * second[i];
    }
    AB_denominator = std::sqrt(squares_sum_A) * std::sqrt(squares_sum_B);
    return AB_numerator / AB_denominator;
}

/* @initial_person - contains person individual vectors(size: 256) from different photos
   @compare_person - with which we compare
*/
float FaceRecognizer::compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person,
                                   float confidence_threshold, float votes_threshold) {
/*    int positive = 0; // positive votes
    if (initial_person.size() < 1) {
        throw "Person vector is empty";
    }
    for (std::vector<float> &photo: initial_person) {
        if (FaceRecognizer::cosSimilarity(photo, compare_person) > confidence_threshold) {
            positive++;
        }
    }
    return (float) positive / initial_person.size()
    *//*if ((float) positive / initial_person.size() >= votes_threshold) {
        return true;
    }
    return false;*/
    if (initial_person.size() != compare_person.size()) {
        throw std::invalid_argument("Both vectors must have the same size");
    }

    const size_t size = initial_person.size();
    float dot = 0;
    float norm1 = 0;
    float norm2 = 0;

    for (size_t i = 0; i < size; i++) {
        dot += initial_person[i] * compare_person[i];
        norm1 += std::pow(initial_person[i], 2);
        norm2 += std::pow(compare_person[i], 2);
    }

    float similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));

    return std::acos(similarity);
}
