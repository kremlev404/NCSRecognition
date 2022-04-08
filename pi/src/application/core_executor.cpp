#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <utility>
#include "core_executor.hpp"

CoreExecutor::CoreExecutor(std::shared_ptr<Classifier> classifier,
                           std::shared_ptr<Detector> face_detector,
                           std::shared_ptr<FaceAligner> aligner,
                           std::shared_ptr<LandmarkDetector> landmark_detector) :
        classifier(std::move(classifier)),
        face_detector(std::move(face_detector)),
        aligner(std::move(aligner)),
        landmark_detector(std::move(landmark_detector)) {

}


void CoreExecutor::initBD(const std::string &db) {
    cv::Mat image, gray, face_image;
    for (const auto &entry: std::filesystem::directory_iterator(db.c_str())) {
        // Get person image
        std::cout << "User photo url: " << entry.path() << std::endl;
        image = cv::imread(entry.path(), cv::IMREAD_COLOR);

        std::vector<float> landmarks;
        cv::Mat mat;

        // Find faces
        std::vector<cv::Rect_<int>> detected_faces = face_detector->detect(image);
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
}

void CoreExecutor::play(const bool gui, const bool flip, cv::VideoCapture capture) {
    cv::Mat image, gray, face_image;
    std::vector<cv::Rect_<int>> faces;
    // Now run webcam stream
    bool need_to_play = true;
    while (need_to_play) {
        std::chrono::high_resolution_clock::time_point t1 =
                std::chrono::high_resolution_clock::now();

        // Get frame and detect faces
        capture >> image;
        if (image.empty()) {
            return;
        }

        if (flip) {
            cv::flip(image, image, 0);
        }
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        faces = face_detector->detect(image);
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
}
