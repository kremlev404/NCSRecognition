#include "face_detector.hpp"

#include <utility>

FaceDetector::FaceDetector(const std::string &xml, const std::string &bin, const std::string &device,
                           float confidence_threshold, int inputWidth, int inputHeight, double scale, cv::Scalar mean,
                           bool swapRB) :
        netSize(cv::Size(inputWidth, inputHeight)),
        confidence_threshold(confidence_threshold),
        scale(scale),
        mean(std::move(mean)),
        swapRB(swapRB) {
    using namespace InferenceEngine;

    Core ie;
    // Reading a network
    this->_network = ie.ReadNetwork(xml, bin);
    this->_network.setBatchSize(1);

    // Get information about topology
    InputsDataMap inputInfo(this->_network.getInputsInfo());
    OutputsDataMap outputInfo(this->_network.getOutputsInfo());

    this->_executable = ie.LoadNetwork(this->_network, device);
    this->_infer_request = this->_executable.CreateInferRequest();
    this->_input = this->_infer_request.GetBlob((*inputInfo.begin()).first);
    this->_output = this->_infer_request.GetBlob((*outputInfo.begin()).first);
}


/*
std::vector<cv::Rect> FaceDetector::detect(cv::Mat image) {

    std::vector<cv::Rect> detected_objects;

    cv::Mat resized_frame;
    cv::resize(image, resized_frame, netSize);

    cv::Mat inputBlob = _input.blobFromImage(resized_frame, scale, netSize, mean, swapRB);

    net.setInput(inputBlob);

    cv::Mat outBlob = _input.forward();

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
}*/
