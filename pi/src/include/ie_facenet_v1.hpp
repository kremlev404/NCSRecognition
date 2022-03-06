#pragma once

#include "classifier_factory.hpp"
#include <inference_engine.hpp>

class IEFacenet_V1: public Classifier {
    private:
        InferenceEngine::ExecutableNetwork _executable;
        InferenceEngine::CNNNetwork _network;
        InferenceEngine::InferRequest _infer_request;
        InferenceEngine::Blob::Ptr _input;
        InferenceEngine::Blob::Ptr _output;
    public:
        IEFacenet_V1(const std::string& xml, const std::string& bin, const std::string& device);
        float distance(const FaceDescriptor& desc1, const FaceDescriptor& desc2) override;
        FaceDescriptor embed(const cv::Mat& face) override;
        ~IEFacenet_V1() override;
};
