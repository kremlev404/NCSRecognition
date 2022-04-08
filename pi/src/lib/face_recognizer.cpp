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
          backEnd(backEnd),
          target(target) {
}

cv::dnn::Net FaceRecognizer::getNet() {
    static auto net = cv::dnn::Net::readFromModelOptimizer(modelPath, configPath);
    static auto is_setted = false;
    if (!is_setted) {
        net.setPreferableBackend(backEnd);
        net.setPreferableTarget(target);
        std::vector available_backends = cv::dnn::getAvailableBackends();
        auto it = available_backends.begin();
        bool target_founded = false;
        while (it != available_backends.end()) {
            auto tg = cv::dnn::getAvailableTargets(it->first);
            if (std::find(tg.begin(), tg.end(), cv::dnn::Target::DNN_TARGET_MYRIAD) != tg.end()) {
                target_founded = true;
                break;
            }
            it++;
        }
        if (!target_founded) {
            throw std::invalid_argument("FaceRecognizer didn't found target");
        } else {
            std::cout << "FaceRecognizer created\n";
        }
        is_setted = true;
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
        throw std::runtime_error("Vectors must have the same size");
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

float FaceRecognizer::compareDescriptors(std::vector<float> &initial_person, std::vector<float> &compare_person) {
    if (initial_person.size() != compare_person.size()) {
        throw std::runtime_error("Both vectors must have the same size");
    }

    return cosSimilarity(initial_person, compare_person);
}
