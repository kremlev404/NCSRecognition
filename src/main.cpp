#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio/videoio_c.h>
#include <stdlib.h>
#include <stdio.h>

#include "face_detector.hpp"
#include "landmarks_detector.hpp"
#include "face_align.hpp"
#include "face_recognition.hpp"
#include "video_assistance.hpp"

using namespace cv;
using namespace std;


static const char* keys =
"{ i images_path      |/study/data/images| path to images }"
"{ m models_path      |/study/data/vinoModels/| path to models  }"
"{ v video_path       |/study/data/video/Elon.mp4| path to models  }"
"{ n numbers          |2| number of images to process }"
"{ o outpath          |/study/data/output/| path to save clips }"
"{ q ? help usage     | <none> | print help message      }";

int main(int argc, char** argv) {

	CommandLineParser parser(argc, argv, keys);
	String home_dir = "";
	std::cout << argc;
	
	if(argc > 1) {
		home_dir = "/home/pi";
	} else {
		home_dir = "/home/kremlev";
	}
	
	if (!parser.check())
	{
		parser.printErrors();
		throw "Parse error";
		return 0;
	}

	String detection_model_path = home_dir + parser.get<String>("models_path")  + "face-detection-retail-0004/FP16/face-detection-retail-0004.xml";
	String detection_config_path = home_dir + parser.get<String>("models_path") + "face-detection-retail-0004/FP16/face-detection-retail-0004.bin";
	String landmarks_model_path = home_dir + parser.get<String>("models_path")  + "landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml";
	String landmarks_config_path = home_dir + parser.get<String>("models_path") + "landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin";
	String recognizer_config_path = home_dir + parser.get<String>("models_path")+ "face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.bin";
	String recognizer_model_path = home_dir + parser.get<String>("models_path") + "face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml";

	String video_path = home_dir + parser.get<String>("video_path");

	String out_path = home_dir + parser.get<String>("outpath"); // path where we will save cropped video clips;

	String images_path = home_dir + parser.get<String>("images_path"); // path where we will save cropped video clips;

	int images_number = parser.get<int>("numbers");

	vector<Mat> images;
	for (int i = 1; i < images_number + 1; i++) {
	    std::string file_name = images_path + "/" + to_string(i) + ".jpg";
        std::cout << file_name << std::endl;
		Mat image = imread(file_name);
		std::cout << image.size() << std::endl;

        images.push_back(image);
        if (images[i - 1].empty()) {
            std::cout << "Image reading error" << std::endl;
        }
	}

	/* Detectors */
	FaceDetector face_detector(detection_model_path, detection_config_path);
	LandmarkDetector landmarks_detector(landmarks_model_path, landmarks_config_path);
	FaceRecognizer recognizer(recognizer_model_path, recognizer_config_path);
	FaceAligner aligner;

	VideoAssistant assistant = VideoAssistant(video_path, images, face_detector, landmarks_detector, recognizer);
	
	//assistant.saveFragments(out_path);

	return 0;
}
