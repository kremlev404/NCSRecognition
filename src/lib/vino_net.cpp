/*
 * Performed by Anton Kremlev
 */

#include <iostream>

#include "vino_net.hpp"

VinoNet::VinoNet(cv::String model_path,
                 cv::String config_path,
                 int input_width,
                 int input_height,
                 double scale,
                 cv::Scalar mean,
                 bool swapRB,
                 int backend,
                 int target) :
        model_path(std::move(model_path)),
        config_path(std::move(config_path)),
        net_size(cv::Size(input_width, input_height)),
        scale(scale), mean(std::move(mean)), swapRB(swapRB),
        backend(backend),
        target(target) {

}

std::shared_ptr<cv::dnn::Net> VinoNet::getNet(const std::string &class_name) {
    auto net = std::make_shared<cv::dnn::Net>(cv::dnn::Net::readFromModelOptimizer(model_path, config_path));

    net->setPreferableBackend(backend);
    net->setPreferableTarget(target);

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
        throw std::invalid_argument(class_name + " didn't found target");
    } else {
        std::cout << class_name << " created\n";
    }

    return net;
}

