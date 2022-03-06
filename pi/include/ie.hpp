# pragma once

#include <inference_engine.hpp>

// Abstract class contains base fields to init IE
class IE {
protected :
    InferenceEngine::ExecutableNetwork _executable;
    InferenceEngine::CNNNetwork _network;
    InferenceEngine::InferRequest _infer_request;
    InferenceEngine::Blob::Ptr _input;
    InferenceEngine::Blob::Ptr _output;
};