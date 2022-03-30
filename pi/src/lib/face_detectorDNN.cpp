#include "face_detectorDNN.hpp"

#include <utility>

FaceDetectorDNN::FaceDetectorDNN(cv::String _modelPath, cv::String _configPath,
                                 float confidence_threshold,
                                 int inputWidth,
                                 int inputHeight,
                                 double scale,
                                 cv::Scalar mean,
                                 bool swapRB,
                                 int backEnd,
                                 int target)
        : modelPath(std::move(_modelPath)),
          backEnd(backEnd),
          target(target),
          configPath(std::move(_configPath)),
          netSize(cv::Size(inputWidth, inputHeight)),
          confidence_threshold(confidence_threshold),
          scale(scale),
          mean(std::move(mean)),
          swapRB(swapRB) {
    std::cout << "FaceDetectorDNN Constructor Called\n" << std::endl;

}

cv::dnn::Net FaceDetectorDNN::getNet() {
    static auto net = cv::dnn::Net::readFromModelOptimizer(modelPath, configPath);
    static auto isSetted = false;
    if (!isSetted) {
        net.setPreferableBackend(backEnd);
        net.setPreferableTarget(target);
        std::cout << "InferenceEngineBackendType: " << cv::dnn::getInferenceEngineBackendType() << " CPU TYPE: "
                  << cv::dnn::getInferenceEngineCPUType() << std::endl;
        std::vector ve = cv::dnn::getAvailableBackends();
        auto[back, pref] = ve[0];
        auto it = ve.begin();
        while (it != ve.end()) {
            std::cout << " it Back: " << it->first << "Target:" << it->second << " " << std::endl;
            auto tg = cv::dnn::getAvailableTargets(it->first);
            auto targetIt = tg.begin();
            while (targetIt != tg.end()) {
                std::cout << "targetIt (0) is CPU, (3) is NCS:" << *targetIt << std::endl;
                targetIt++;
            }
            it++;
        }
        isSetted = true;
    }
    return net;
}

std::vector<cv::Rect> FaceDetectorDNN::detect(const cv::Mat &image) {
    //cv::dnn::resetMyriadDevice();
    auto net = getNet();

    std::cout << "FaceDetectorDNN detect\n" << std::endl;

    std::vector<cv::Rect> detected_objects;

    cv::Mat resized_frame;
    cv::resize(image, resized_frame, netSize);

    cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);

    net.setInput(inputBlob);

    cv::Mat outBlob = net.forward();
    std::cout << "FaceDetectorDNN forwarded\n" << std::endl;
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


/*
 * FaceDetector::FaceDetector(const std::string &xml, const std::string &bin, const std::string &device,
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

    outputInfo.begin()->second->getDims();

    this->_executable = ie.LoadNetwork(this->_network, device);
    this->_infer_request = this->_executable.CreateInferRequest();
    this->_input = this->_infer_request.GetBlob((*inputInfo.begin()).first);
    this->_output = this->_infer_request.GetBlob((*outputInfo.begin()).first);
    std::cout << "FaceDetector Created\n" << std::endl;
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& image) {

    std::vector<cv::Rect> detected_objects;
    std::cout << "FaceDetector::Detect\n" << std::endl;

    using namespace InferenceEngine;

    const cv::Size expectedImageSize = cv::Size(160, 160);
    cv::Mat floatImage;

    if (image.size() != expectedImageSize) {
        cv::resize(image, floatImage, expectedImageSize);
    }

    cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);
    floatImage.convertTo(floatImage, CV_32FC3);

    // Prepare data
    auto data = this->_input->buffer().as<float*>();

    size_t num_channels = this->_input->getTensorDesc().getDims()[1];
    size_t width = this->_input->getTensorDesc().getDims()[2];
    size_t height = this->_input->getTensorDesc().getDims()[3];
    size_t image_size = width * height;
    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            for (size_t ch = 0; ch < num_channels; ch++) {
                data[ch * image_size + h * width + w] =
                        floatImage.at<cv::Vec3f>(h, w)[num_channels + ch];
            }
        }
    }

    this->_infer_request.Infer();

    // get output
    const auto predictions = this->_output->buffer().as<float *>();
    size_t detection_size = this->_output->getTensorDesc().getDims().at(2);

    std::vector<float> result;
    result.reserve(detection_size);

    for (size_t id = 0; id < detection_size; id++) {
        result.push_back(predictions[id]);
    }

    float confidence = 0;
    float cls = 0;
    float id = 0;
    size_t founded_faces = detection_size / 7;
    std::cout << "FaceDetector::Detect (x:y) " << founded_faces <<" "<< detection_size << std::endl;

    for (int i = 0; i < detection_size; i++)
    {
        std::cout << "FaceDetector::Detect (x:y) " << i << std::endl;

        confidence = predictions[i * 7 + 2];
        cls = predictions[i * 7 + 1];
        id = predictions[i * 7];
        if (id >= 0 && confidence > confidence_threshold && cls <= 1)
        {
            detected_objects.emplace_back(predictions[i * 7 + 3] * width, predictions[i * 7 + 4] * height,
                               (predictions[i * 7 + 5] - predictions[i * 7 + 3]) * width,
                               (predictions[i * 7 + 6] - predictions[i * 7 + 4]) * height);
            std::cout << "FaceDetector::Detect (x:y) " << detected_objects.begin()->x << " : " << detected_objects.end()->y << std::endl;

        }
    }

    return detected_objects;
}
*/
