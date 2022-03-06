#include "face_detector.hpp"

FaceDetector::FaceDetector(cv::String _modelPath, cv::String _configPath,
	float confidence_threshold, 
	int inputWidth,
	int inputHeight,
	double scale,
	cv::Scalar mean,
	bool swapRB,
	int backEnd,
	int target)
	:modelPath(_modelPath),
	configPath(_configPath),
	netSize(cv::Size(inputWidth, inputHeight)),
	confidence_threshold(confidence_threshold),
	scale(scale),
	mean(mean),
	swapRB(swapRB)
{
	std::cout << cv::dnn::getInferenceEngineVPUType();
	backEnd = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
    target = cv::dnn::DNN_TARGET_MYRIAD;

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
	
	net = cv::dnn::readNetFromModelOptimizer(modelPath, configPath);
}


std::vector<cv::Rect> FaceDetector::detect(cv::Mat image) {

	std::vector<cv::Rect> detected_objects;
	
	cv::Mat resized_frame;
	cv::resize(image, resized_frame, netSize);
	std::cout <<"tut";

	std::cout << cv::dnn::getInferenceEngineVPUType();
	//cv::dnn::setInferenceEngineBackendType(CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X);
	cv::Mat inputBlob = cv::dnn::blobFromImage(resized_frame, scale, netSize, mean, swapRB);
	std::cout <<"tut";

	net.setInput(inputBlob);
		std::cout <<"tut1";

	cv::Mat outBlob = net.forward();
		std::cout <<"tut2";

	cv::Mat detection_as_mat(outBlob.size[2], outBlob.size[3], CV_32F, outBlob.ptr<float>());


	for (int i = 0; i < detection_as_mat.rows; i++)
	{
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
