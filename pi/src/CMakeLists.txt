# BUILD headers
include_directories("include")
include_directories("${OpenCV_INCLUDE_DIRS}")
include_directories("${IE_DIR}/include")

# LINK
link_directories("${IE_DIR}/lib/armv7l")
link_directories("${IE_DIR}/lib/intel64")

set(IE_SHARED_LIBS libinference_engine.so)

# MAKE LIBRARY
set(SOURCES lib/factory.cpp lib/ie_classifier.cpp lib/face_detectorDNN.cpp ../include/ie.hpp include/face_detector.hpp lib/face_detector.cpp ../include/detector.hpp ../include/classifier.hpp)
add_library(CPPClassificator SHARED ${SOURCES})
target_link_libraries(CPPClassificator ${OpenCV_LIBS} ${IE_SHARED_LIBS})

# MAKE Application
set(SOURCES application/main.cpp)
add_executable(NCSRecognition ${SOURCES})
target_link_libraries(NCSRecognition CPPClassificator ${OpenCV_LIBS} ${IE_SHARED_LIBS}  "stdc++fs")

# MAKE Install
install(TARGETS CPPClassificator NCSRecognition
    DESTINATION ${PROJECT_SOURCE_DIR}/install/bin)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION ${PROJECT_SOURCE_DIR}/install)
