/*
 * Performed by Anton Kremlev
 */

#include "face_detector.hpp"

#include <utility>

FaceDetector::FaceDetector(cv::String model_path, cv::String config_path,
                           float confidence_threshold,
                           int input_width,
                           int input_height,
                           double scale)
        : VinoNet(
        std::move(model_path),
        std::move(config_path),
        input_width,
        input_height,
        scale),
          confidence_threshold(confidence_threshold) {
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat &image) {
    //cv::dnn::resetMyriadDevice();
    static auto net = getNet("FaceDetector");

    std::vector<cv::Rect> detected_objects;

    cv::Mat resized_frame;
    cv::resize(image, resized_frame, net_size);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, net_size, mean, swapRB);

    net->setInput(inputBlob);

    cv::Mat outBlob = net->forward();
    cv::Mat detection_as_mat(outBlob.size[2], outBlob.size[3], CV_32F, outBlob.ptr<float>());

    for (int i = 0; i < detection_as_mat.rows; i++) {
        float cur_confidence = detection_as_mat.at<float>(i, 2);
        int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * image.cols);
        int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * image.rows);
        int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * image.cols);
        int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * image.rows);
        cv::Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

        if (cur_confidence < confidence_threshold)
            continue;

        if (cur_rect.empty())
            continue;

        cur_rect = cur_rect & cv::Rect(cv::Point(), image.size());
        detected_objects.push_back(cur_rect);
    }
    return detected_objects;
}