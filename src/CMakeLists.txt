# BUILD headers
include_directories("include")
include_directories("${OpenCV_INCLUDE_DIRS}")
include_directories("${IE_DIR}/include")

# LINK
link_directories("${IE_DIR}/lib/armv7l")
link_directories("${IE_DIR}/lib/intel64")

set(IE_SHARED_LIBS libinference_engine.so)

# MAKE library
set(SOURCES
        ../include/detector.hpp
        ../include/classifier.hpp
        include/data/face_data.hpp
        lib/factory.cpp
        lib/face_detector.cpp
        include/face_recognizer.hpp
        lib/face_recognizer.cpp
        include/landmarks_detector.hpp
        lib/landmarks_detector.cpp
        include/face_aligner.hpp
        lib/face_aligner.cpp
        include/face_detector_cascade.hpp
        lib/face_detector_cascade.cpp
        include/core_executor.hpp
        lib/vino_net.cpp
        include/vino_net.hpp
        lib/timer.cpp
        include/timer.hpp include/firebase_interactor.hpp lib/firebase_interactor.cpp include/data/person_period_data.hpp)

add_library(CPPClassificator SHARED ${SOURCES})
target_link_libraries(CPPClassificator ${OpenCV_LIBS} ${IE_SHARED_LIBS})

# MAKE Application
set(SOURCES application/main.cpp application/core_executor.cpp)
add_executable(NCSRecognition ${SOURCES})
target_link_libraries(NCSRecognition CPPClassificator ${OpenCV_LIBS} ${IE_SHARED_LIBS}  "stdc++fs")

# MAKE Install
install(TARGETS CPPClassificator NCSRecognition
    DESTINATION ${PROJECT_SOURCE_DIR}/install/bin)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION ${PROJECT_SOURCE_DIR}/install)
