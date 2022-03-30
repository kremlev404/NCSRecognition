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
    ie.SetConfig({}, device);
    this->_network = ie.ReadNetwork(xml, bin);
    this->_network.setBatchSize(1);

    // Get information about topology
    InputsDataMap inputInfo(this->_network.getInputsInfo());
    OutputsDataMap outputsInfo(this->_network.getOutputsInfo());
    DataPtr outputInfo;
    outputInfo  = outputsInfo.begin()->second;

    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();
    maxProposalCount  = outputDims[2];
    objectSize        = outputDims[3];
    outputName = outputInfo->getName();

    this->_executable = ie.LoadNetwork(this->_network, device);
    std::cout << "FaceDetector Created\n" << std::endl;

    this->_infer_request = this->_executable.CreateInferRequest();
    this->_input = this->_infer_request.GetBlob((*inputInfo.begin()).first);
    this->_output = this->_infer_request.GetBlob((*outputsInfo.begin()).first);
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

    size_t width = this->_input->getTensorDesc().getDims()[2];
    size_t height = this->_input->getTensorDesc().getDims()[3];
// --------------------------- Step 6. Prepare input
    // --------------------------------------------------------
    /** Collect images data ptrs **/
    std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
    std::vector<size_t> imageWidths, imageHeights;

/*    FormatReader::ReaderPtr reader(i.c_str());
    if (reader.get() == nullptr) {
        slog::warn << "Image " + i + " cannot be read!" << slog::endl;
        continue;
    }
    *//** Store image data **//*
    std::shared_ptr<unsigned char> originalData(reader->getData());
    std::shared_ptr<unsigned char> data(reader->getData(image->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
    if (data.get() != nullptr) {
        originalImagesData.push_back(originalData);
        imagesData.push_back(data);
        imageWidths.push_back(reader->width());
        imageHeights.push_back(reader->height());
    }*/

    auto batchSize = std::min(_network.getBatchSize(), _input->size());
    MemoryBlob::Ptr mimage = as<MemoryBlob>(_input);
    if (!mimage) {
        std::cout << "We expect image blob to be inherited from MemoryBlob, but "
                     "by fact we were not able "
                     "to cast imageInput to MemoryBlob"
                  << std::endl;
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = mimage->wmap();

    size_t num_channels = mimage->getTensorDesc().getDims()[1];
    size_t image_size = mimage->getTensorDesc().getDims()[3] * mimage->getTensorDesc().getDims()[2];

    unsigned char* data = minputHolder.as<unsigned char*>();
    /** Iterate over all input images limited by batch size  **/
    for (size_t image_id = 0; image_id < std::min(_input->size(), batchSize); ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** Iterate over all channels **/
            for (size_t ch = 0; ch < num_channels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in
                 * bytes            **/
               // data[image_id * image_size * num_channels + ch * image_size + pid] = _input.at(image_id).get()[pid * num_channels + ch];
            }
        }
    }

    this->_infer_request.Infer();

    const Blob::Ptr output_blob = _infer_request.GetBlob(outputName);
    MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
    if (!moutput) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                               "but by fact we were not able to cast output to MemoryBlob");
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto moutputHolder = moutput->rmap();
    const float* detection = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++)
    {
        auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
        if (image_id < 0) {
            break;
        }

        float confidence = detection[curProposal * objectSize + 2];
        auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
        auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * width );
        auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * height);
        auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * width );
        auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * height);

        std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence << "    (" << xmin << "," << ymin << ")-(" << xmax << ","
                  << ymax << ")"
                  << " batch id : " << image_id;

        if (confidence > 0.5) {
            /** Drawing only objects with >50% probability **/
            std::cout << " WILL BE PRINTED!" << confidence;

            detected_objects.emplace_back(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
            std::cout << "FaceDetector::Detect (x:y) " << xmin << " : "
                      << ymin << std::endl;
        }
    }

    return detected_objects;
}