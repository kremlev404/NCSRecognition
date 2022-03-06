#include <filesystem>
#include <iostream>
#include <memory>
#include <map>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "classifier.hpp"

static const cv::String keys =
        "{user_name      |kremlev| name of system user        }"
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

    if (!parser.check()) {
        parser.printErrors();
        throw "Parse error";
        return 0;
    }

    if (parser.get<bool>("help") == true) {
        std::cout << keys;
        return 0;
    }

    std::string home_dir = "/home/" + parser.get<std::string>("user_name");
    std::string xml, bin, detector, db;

    if (parser.get<bool>("args_include") == true) {
        xml = parser.get<std::string>("xml");
        bin = parser.get<std::string>("bin");
        detector = parser.get<std::string>("detector");
        db = parser.get<std::string>("db");
    } else {
        xml = bin = detector = db = home_dir;
        xml += "/study/data/face-reidentification-retail-0095.xml";
        bin += "/study/data/face-reidentification-retail-0095.bin";
        detector += "/study/data/haarcascade_frontalcatface.xml";
        db += "/study/data/db/";
    }

    const std::string device = parser.get<std::string>("device");
    const bool gui = parser.get<bool>("gui");
    const bool flip = parser.get<bool>("flip");
    const int width = parser.get<int>("width");
    const int height = parser.get<int>("height");

    cv::VideoCapture capture(home_dir + "/study/data/video/me.mp4");
    if (!capture.isOpened()) {
        throw new std::runtime_error("Couldn't open a video stream");
    }

    std::cout << "Device: " << device << std::endl;
    std::cout << "XML: " << xml << std::endl;
    std::cout << "BIN: " << bin << std::endl;
    std::cout << "Face detector: " << detector << std::endl;
    std::cout << "People: " << db << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "gui: " << gui << std::endl;

    if (gui == true) {
        cv::namedWindow("NCSRecognition");
    }

    // Load face detector
    cv::CascadeClassifier cascade;
    cascade.load(detector);

    const std::shared_ptr<Classifier> classifier = build_classifier(
            ClassifierType::IE_Facenet_V1, xml, bin, device);

    std::vector<cv::Rect> faces;

    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat image, gray, face_image;

    // Find all people in the directory
    std::map<std::string, std::vector<float>> people;
    for (const auto &entry: std::filesystem::directory_iterator(db.c_str())) {
        // Get person image
        image = cv::imread(entry.path(), cv::IMREAD_COLOR);

        // Find faces
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cascade.detectMultiScale(gray, faces, 1.5, 5, 0, cv::Size(150, 150));

        // There must be one face per image
        face_image = image(faces[0]);

        // Get and save embedding for a face
        // The library expects BGR image
        cv::resize(face_image, face_image, cv::Size(160, 160));
        std::vector<float> reference = classifier->embed(face_image);
        std::cout << entry.path() << std::endl;
        for (float &number: reference) {
            std::cout << number << ",";
        }

        std::cout << std::endl;
        people.insert(std::pair<std::string, std::vector<float>>(entry.path().filename(), reference));
    }

    // Now run webcam stream
    while (true) {
        std::chrono::high_resolution_clock::time_point t1 =
                std::chrono::high_resolution_clock::now();

        // Get frame and detect faces
        capture >> image;
        if (flip) {
            cv::flip(image, image, 0);
        }

        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cascade.detectMultiScale(gray, faces, 1.5, 5, 0, cv::Size(150, 150));

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
            // Get ROI
            face_image = image(face);

            // Get embedding
            cv::resize(face_image, face_image, cv::Size(160, 160));
            std::vector<float> result = classifier->embed(face_image);
            cv::rectangle(image, face, cv::Scalar(255, 0, 255));

            // Find it's across saved people
            float minDistance = 100;
            std::string minKey;
            for (const std::pair<std::string, std::vector<float>> &pair: people) {
                float distance = classifier->distance(pair.second, result);
                if (distance < minDistance) {
                    minDistance = distance;
                    minKey = pair.first;
                }
            }

            // Approximate threshold
            if (minDistance > 0.5) {
                cv::putText(image, "unknown", cv::Point(face.tl()),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0, 0, 255));
            } else {
                std::string text =
                        minKey + std::string(": ") + std::to_string(minDistance);
                cv::putText(image, text, cv::Point(face.tl()),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0, 0, 255));

                if (gui != true) {
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

        if (gui == true) {
            imshow("NCSRecognition", image);
            const int waitKey = cv::waitKey(20);
            if (waitKey == 27) {
                break;
            }
        }
    }

    capture.release();
}
