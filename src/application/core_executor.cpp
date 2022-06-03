/*
 * Performed by Anton Kremlev
 */

#include <filesystem>
#include <fstream>
#include <utility>

#include <opencv2/core/mat.hpp>

#include "core_executor.hpp"
#include "face_recognizer.hpp"
#include "timer.hpp"
#include "firebase_interactor.hpp"

CoreExecutor::CoreExecutor(std::shared_ptr<IClassifier> classifier,
                           std::shared_ptr<IDetector> face_detector,
                           std::shared_ptr<FaceAligner> aligner,
                           std::shared_ptr<LandmarkDetector> landmark_detector,
                           std::shared_ptr<IGPIO> gpio_controller,
                           const int &update_period,
                           const bool &to_gray_filter) :
        classifier(std::move(classifier)),
        face_detector(std::move(face_detector)),
        aligner(std::move(aligner)),
        landmark_detector(std::move(landmark_detector)),
        gpio_controller(std::move(gpio_controller)),
        firebase_interactor(std::make_unique<FirebaseInteractor>(update_period)),
        timer(std::make_unique<Timer>(update_period)),
        use_gray_filter(to_gray_filter) {
    std::cout << "CoreExecutor created\n";
}

std::vector<float> CoreExecutor::getEmbed(const std::string &path_to_image) {
    cv::Mat image;
    cv::Mat gray;

    image = cv::imread(path_to_image, cv::IMREAD_COLOR);
    cv::resize(image, image, window_size);
    std::vector<float> landmarks;

    std::vector<cv::Rect_<int>> detected_faces;
    if (use_gray_filter) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        detected_faces = face_detector->detect(gray);
    } else {
        detected_faces = face_detector->detect(image);
    }

    if (detected_faces.empty())
        return {};

    cv::Rect face_rect = detected_faces[0];
    landmarks = landmark_detector->detect(image(face_rect));
    if (landmarks.empty())
        return {};

    cv::Mat transformedFace = aligner->align(image(face_rect), landmarks);
    cv::resize(transformedFace, transformedFace, classifier->getInputSize());

    return classifier->embed(transformedFace);
}


void CoreExecutor::initBD(const std::string &db) {
    cv::Mat image;
    const std::string dummy_folder = "..";

    using r_iterator = std::filesystem::recursive_directory_iterator;
    std::filesystem::path current_folder = dummy_folder;
    auto person_descriptors = std::vector<FaceData>();

    for (r_iterator i(db.c_str()), end; i != end; ++i) {
        auto name = i->path().filename();
        if (!is_directory(i->path())) {
            auto descriptor = getEmbed(i->path());
            if (descriptor.empty()) {
                continue;
            }
            person_descriptors.push_back(FaceData{name, descriptor});
        } else {
            if (current_folder != name) {
                if (current_folder != dummy_folder) {
                    peoples.insert(std::pair(current_folder, person_descriptors));
                }
                current_folder = name;
                person_descriptors.clear();
            }
        }
    }


    if (!person_descriptors.empty()) {
        peoples.insert(std::pair(current_folder, person_descriptors));
        person_descriptors.clear();
    }

    std::cout << "DB of users scanned" << std::endl;
    std::cout << peoples.size() << "\n";

    for (const auto &[key, value]: peoples) {
        for (const auto &it: value) {
            std::cout << "Person: " << key << " FileName: " << it.file_name;
            float avg = 0;
            //std::reduce doesn't exist in std in gnu 8.3 witch used in raspberry
            for (const auto &descriptor_value: it.descriptor) {
                avg += descriptor_value;
            }
            std::cout << " AVG: " << avg / static_cast<float>( it.descriptor.size()) << " Desc: ";
            for (const auto &i: it.descriptor) {
                std::cout << i << ",";
            }

            std::cout << std::endl;
        }
    }
}

void CoreExecutor::play(const bool &gui, const bool &flip, const std::shared_ptr<cv::VideoCapture> &capture) {
    timer->start([capture = firebase_interactor.get()] { capture->send_to_firebase(); });

    cv::Mat image;
    cv::Mat gray;
    std::vector<cv::Rect_<int>> faces;
    int32_t frame_counter = 0;
    float time_counter = 0;
    bool need_to_play = true;

    std::fstream f;
    std::string file_path = "../../data/statistic/statisticH";

    if (use_gray_filter) {
        file_path += "G.txt";
    } else {
        file_path += ".txt";
    }

    f.open(file_path, std::fstream::in | std::fstream::out);
    if (!f.is_open()) {
        std::cout << "error!";
        throw std::runtime_error("file error");
    }
    std::vector<Stats> statistic;

    while (need_to_play) {
        std::chrono::high_resolution_clock::time_point t1 =
                std::chrono::high_resolution_clock::now();

        *capture >> image;
        if (image.empty()) {
            for (const auto &stat: statistic) {
                f << stat.frame << "\t" << stat.id << "\t" << stat.prob << "\t" << stat.fps << std::endl;
            }

            f.close();
            reset();
            return;
        }
        frame_counter++;
        if (flip) {
            cv::flip(image, image, 0);
        }
        cv::resize(image, image, window_size);

        if (use_gray_filter) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            faces = face_detector->detect(gray);
        } else {
            faces = face_detector->detect(image);
        }
        if (faces.empty()) {
            gpio_controller->ledOff(LedOutput::green_led);
            gpio_controller->ledOff(LedOutput::red_led);
        }
        for (cv::Rect &face: faces) {
            for (const cv::Rect &another_face: faces) {
                if (face.x > another_face.x && face.y > another_face.y
                    && face.x + face.width < another_face.x + another_face.width
                    && face.y + face.height < another_face.y + another_face.height) {
                    continue;
                }
            }

            std::vector<float> landmarks = landmark_detector->detect(image(face));
            cv::Mat transformed_face = aligner->align(image(face), landmarks);

            cv::resize(transformed_face, transformed_face, classifier->getInputSize());
            std::vector<float> result = classifier->embed(transformed_face);

            float max_distance = -1;
            std::string max_key;

            for (const auto &[person_name, person_data]: peoples) {
                for (const auto &face_data: person_data) {
                    float distance = classifier->compareDescriptors(face_data.descriptor, result);

                    if (distance > max_distance) {
                        max_distance = distance;
                        max_key = person_name;
                    }
                }
            }

            if (max_distance < 0.5) {
                gpio_controller->ledOff(LedOutput::green_led);
                gpio_controller->ledOn(LedOutput::red_led);

                cv::rectangle(image, face, unknown_color);
                cv::putText(image, "unknown", cv::Point(face.tl()),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, unknown_color);

            } else {
                gpio_controller->ledOff(LedOutput::red_led);
                gpio_controller->ledOn(LedOutput::green_led);

                firebase_interactor->push(max_key, max_distance);
                cv::rectangle(image, face, known_color);
                std::string text =
                        max_key + std::string(": ") + std::to_string(max_distance);
                cv::putText(image, text, cv::Point(face.tl()),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, known_color);


                if (!gui) {
                    std::cout << "Found -> " << text << std::endl;
                }
            }
            statistic.emplace_back(frame_counter, max_key, max_distance, getAvgFps());
        }

        // Compute FPS
        std::chrono::high_resolution_clock::time_point t2 =
                std::chrono::high_resolution_clock::now();
        auto difference =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        time_counter += static_cast<float>(difference) / 1000;

        avg_fps = static_cast<float>(frame_counter) / (time_counter);
        cv::putText(image, "FPS: " + std::to_string(1000 / difference),
                    cv::Point(1, 17),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, color
        );
        cv::putText(image, "AVG FPS: " + std::to_string(avg_fps),
                    cv::Point(1, 42),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, color
        );
        cv::putText(image, "Frame:  " + std::to_string(frame_counter),
                    cv::Point(1, 600),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 3.0, unknown_color
        );

        if (gui) {
            cv::imshow("NCSRecognition", image);
            cv::waitKey(1);
            try {
                if (cv::getWindowImageRect("NCSRecognition").x == -1) {
                    need_to_play = false;
                    std::cout << "X was pressed\n";
                }
            } catch (cv::Exception &e) {
                reset();
                std::cout << "X was pressed\n";
                return;
            }
        }
    }

    reset();
}

float CoreExecutor::getAvgFps() const {
    return avg_fps;
}

void CoreExecutor::reset() {
    timer->stop();
    gpio_controller->ledOff(LedOutput::green_led);
    gpio_controller->ledOff(LedOutput::red_led);
}
