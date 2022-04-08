/*
 * Performed by Anton Kremlev
 */

#include "face_recognizer.hpp"

std::vector<float> FaceRecognizer::embed(const cv::Mat &image) {
    static auto net = getNet("FaceRecognizer");
    std::vector<cv::Rect> detected_objects;

    cv::Mat resized_frame;
    cv::resize(image, resized_frame, net_size);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, net_size, mean, swapRB);
    net->setInput(inputBlob);

    cv::Mat outBlob = net->forward();
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
