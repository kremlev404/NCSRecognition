#include <iostream>
#include <memory>
#include <map>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "classifier.hpp"
#include "detector.hpp"
#include "face_aligner.hpp"
#include "landmarks_detector.hpp"
#include "core_executor.hpp"

static const cv::String keys =
        "{user_name      |pi| name of system user        }"
        "{args_include   |false| use custom config               }"
        "{device         |MYRIAD| backend device (CPU, MYRIAD)}"
        "{xml            |<none>| path to model definition    }"
        "{bin            |<none>| path to model weights       }"
        "{detector       |<none>| path to face detector      }"
        "{db             |<none>| path to reference people    }"
        "{width          |640| stream width                   }"
        "{height         |480| stream height                  }"
        "{flip           |false| flip stream images           }"
        "{gui            |true| show gui                       }"
        "{help           |false| show gui                       }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);

    if (!parser.check()) {
        parser.printErrors();
        throw std::invalid_argument("Parse error");
    }

    if (parser.get<bool>("help")) {
        std::cout << keys;
        return 0;
    }

    auto detector_type = DetectorType::face_detection_retail_0004;
    auto classifier_type = ClassifierType::face_reidentification_retail_0095;

    std::string home_dir = "/home/" + parser.get<std::string>("user_name");
    std::string recognition_xml, recognition_bin, detector_xml, detector_bin, landmark_xml, landmark_bin, db;

    if (parser.get<bool>("args_include")) {
        recognition_xml = parser.get<std::string>("recognition_xml");
        recognition_bin = parser.get<std::string>("recognition_bin");
        detector_xml = parser.get<std::string>("detector_xml");
        db = parser.get<std::string>("db");
    } else {
        recognition_xml = recognition_bin = detector_xml = detector_bin = landmark_xml = landmark_bin = db = home_dir;
        recognition_xml += "/study/data/face-reidentification-retail-0095.xml";
        recognition_bin += "/study/data/face-reidentification-retail-0095.bin";
        landmark_xml += "/study/data/landmarks-regression-retail-0009.xml";
        landmark_bin += "/study/data/landmarks-regression-retail-0009.bin";
        switch (detector_type) {
            case DetectorType::face_detection_retail_0004: {
                detector_xml += "/study/data/face-detection-retail-0004/FP16/face-detection-retail-0004.xml";
                detector_bin += "/study/data/face-detection-retail-0004/FP16/face-detection-retail-0004.bin";
                break;
            }
            case DetectorType::face_detection_retail_0001: {
                detector_xml += "/study/data/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
                detector_bin += "/study/data/face-detection-adas-0001/FP16/face-detection-adas-0001.bin";
                break;
            }
            case DetectorType::haar_cascade: {
                detector_xml += "/study/data/haarcascade_frontalcatface.xml";
                break;
            }
            default: {
                return -1;
            }
        }
        db += "/study/data/db/";
    }

    const auto device = parser.get<std::string>("device");
    const bool gui = parser.get<bool>("gui");
    const bool flip = parser.get<bool>("flip");
    const int width = parser.get<int>("width");
    const int height = parser.get<int>("height");

    cv::VideoCapture capture(home_dir + "/study/data/video/me.mp4");
    if (!capture.isOpened()) {
        throw std::runtime_error("Couldn't open a video stream");
    }
    if (gui) {
        cv::namedWindow("NCSRecognition");
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    std::cout << "Device: " << device << std::endl;
    std::cout << "Recognize XML: " << recognition_xml << std::endl;
    std::cout << "Recognize BIN: " << recognition_bin << std::endl;
    std::cout << "Detector_xml: " << detector_xml << std::endl;
    std::cout << "Detector_bin: " << detector_bin << std::endl;
    std::cout << "People: " << db << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "gui: " << gui << std::endl;

    const std::shared_ptr<Classifier> classifier = build_classifier(classifier_type, recognition_xml, recognition_bin,
                                                                    device);
    auto face_detector = build_detector(detector_type, detector_xml, detector_bin);
    auto aligner = std::make_shared<FaceAligner>();
    auto landmark_detector = std::make_shared<LandmarkDetector>(landmark_xml, landmark_bin);

    auto core_executor = std::make_unique<CoreExecutor>(classifier, face_detector, aligner, landmark_detector);

    core_executor->initBD(db);
    core_executor->play(gui, flip, capture);
    std::cout << "AVG FPS: " << core_executor->getAvgFps() << std::endl;
    capture.release();
    cv::destroyAllWindows();
}
