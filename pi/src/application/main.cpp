#include <filesystem>
#include <iostream>
#include <memory>
#include <map>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "classifier.hpp"
#include "detector.hpp"
#include "face_aligner.hpp"
#include "landmarks_detector.hpp"

static const cv::String keys =
        "{user_name      |pi| name of system user        }"
        "{args_include   |false| use custom config               }"
        "{device         |MYRIAD| backend device (CPU, MYRIAD)}"
        "{xml            |<none>| path to model definition    }"
        "{bin            |<none>| path to model weights       }"
        "{detector       |<nonde>| path to face detector      }"
        "{db             |<none>| path to reference people    }"
        "{width          |640| stream width                   }"
        "{height         |480| stream height                  }"
        "{flip           |false| flip stream images           }"
        "{gui            |true| show gui                       }"
        "{help           |false| show gui                       }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    auto detector_type = DetectorType::face_detection_retail_0004;
    auto classifier_type = ClassifierType::face_reidentification_retail_0095;
    if (!parser.check()) {
        parser.printErrors();
        throw std::invalid_argument("Parse error");
        return 0;
    }

    if (parser.get<bool>("help")) {
        std::cout << keys;
        return 0;
    }

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
        // Find faces
        switch (detector_type) {
            case DetectorType::face_detection_retail_0004: {
                //
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

    std::cout << "Device: " << device << std::endl;
    std::cout << "Recognize XML: " << recognition_xml << std::endl;
    std::cout << "Recognize BIN: " << recognition_bin << std::endl;
    std::cout << "Detector_xml: " << detector_xml << std::endl;
    std::cout << "Detector_bin: " << detector_bin << std::endl;
    std::cout << "People: " << db << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "gui: " << gui << std::endl;

    if (gui) {
        cv::namedWindow("NCSRecognition");
    }

    // Load face detector_xml
    // cv::CascadeClassifier cascade;
    // cascade.load(detector_xml);

    const std::shared_ptr<Classifier> classifier = build_classifier(classifier_type, recognition_xml, recognition_bin,
                                                                    device);
    auto vino_detector = build_detector(detector_type, detector_xml, detector_bin);
    auto aligner = std::make_unique<FaceAligner>();
    auto landmark_detector = std::make_unique<LandmarkDetector>(landmark_xml, landmark_bin);

    std::vector<cv::Rect> faces;

    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat image, gray, face_image;

    // Find all people in the directory
    std::map<std::string, std::vector<float>> people;
    for (const auto &entry: std::filesystem::directory_iterator(db.c_str())) {
        // Get person image
        std::cout << "User photo url: " << entry.path() << std::endl;
        image = cv::imread(entry.path(), cv::IMREAD_COLOR);

        std::vector<float> landmarks;
        cv::Mat mat;

        // Find faces
        std::vector<cv::Rect_<int>> detected_faces = vino_detector->detect(image);
        cv::Rect face_rect = detected_faces[0];
        landmarks = landmark_detector->detect(image(face_rect));
        cv::Mat transformedFace = aligner->align(image(face_rect), landmarks);;

        // There must be one face per image
        face_image = transformedFace;// image(faces[0]);

        // Get and save embedding for a face
        // The library expects BGR image
        cv::resize(face_image, face_image, cv::Size(160, 160));
        std::vector<float> reference = classifier->embed(face_image);
        std::cout << entry.path() << " descriptor: ";

        for (float &number: reference) {
            std::cout << number << ",";
        }

        std::cout << std::endl;
        people.insert(std::pair<std::string, std::vector<float>>(entry.path().filename(), reference));
        std::cout << "DB of users scanned" << std::endl;
    }

    // Now run webcam stream
    bool need_to_play = true;
    while (need_to_play) {
        std::chrono::high_resolution_clock::time_point t1 =
                std::chrono::high_resolution_clock::now();

        // Get frame and detect faces
        capture >> image;
        if (flip) {
            cv::flip(image, image, 0);
        }
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        //cascade.detectMultiScale(gray, faces, 1.5, 5, 0, cv::Size(150, 150));
        faces = vino_detector->detect(image);
        for (cv::Rect &face: faces) {
            bool ignore = false;
            for (cv::Rect &another_face: faces) {
                if (face.x > another_face.x && face.y > another_face.y
                    && face.x + face.width < another_face.x + another_face.width
                    && face.y + face.height < another_face.y + another_face.height) {
                    ignore = true;
                }
            }
            if (ignore) {
                continue;
            }
            std::vector<float> landmarks = landmark_detector->detect(image(face));
            cv::Mat transformedFace = aligner->align(image(face), landmarks);

            // Get ROI
            face_image = transformedFace;// image(face);
            // Get embedding
            cv::resize(face_image, face_image, cv::Size(160, 160));
            std::vector<float> result = classifier->embed(face_image);
            cv::rectangle(image, face, cv::Scalar(255, 0, 255));
            // Find it's across saved people
            float maxDistance = -1;
            std::string minKey;
            for (std::pair<std::string, std::vector<float>> pair: people) {
                float distance = classifier->compareDescriptors(pair.second, result);
                //std::cout << "[MAIN] Distance: " << distance << std::endl;
                if (distance > maxDistance) {
                    maxDistance = distance;
                    minKey = pair.first;
                }
            }

            // Approximate threshold
            if (maxDistance < 0.5) {
                cv::putText(image, "unknown", cv::Point(face.tl()),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0, 0, 255));
            } else {
                std::string text =
                        minKey + std::string(": ") + std::to_string(maxDistance);
                cv::putText(image, text, cv::Point(face.tl()),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0, 0, 255));

                if (!gui) {
                    std::cout << "Found -> " << text << std::endl;
                }
            }
        }

        // Compute FPS
        std::chrono::high_resolution_clock::time_point t2 =
                std::chrono::high_resolution_clock::now();
        auto difference =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        cv::putText(image, std::to_string(1000 / difference), cv::Point(50, 50),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(0, 255, 255)
        );

        if (gui) {
            cv::imshow("NCSRecognition", image);
            cv::waitKey(1);
            if (cv::getWindowImageRect("NCSRecognition").x == -1) {
                std::cout << "X was pressed";
                need_to_play = false;
            }
        }
    }

    capture.release();
    cv::destroyAllWindows();
}
