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
        ../include/idetector.hpp
        ../include/iclassifier.hpp
        ../include/igpio.hpp
        include/data/face_data.hpp
        include/data/person_period_data.hpp
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
        include/timer.hpp
        include/firebase_interactor.hpp
        lib/firebase_interactor.cpp
        )

if (IS_RASPBERRY)
    list(APPEND SOURCES
            ../pi/raspberry/gpio.cpp
            ../pi/raspberry/gpio.hpp
            ../pi/raspberry/gpio_factory.cpp
            )

else ()
    list(APPEND SOURCES
            ../pi/x86/gpio.hpp
            ../pi/x86/gpio_factory.cpp
            )
endif ()

add_library(NCSLib SHARED ${SOURCES})
target_link_libraries(NCSLib ${OpenCV_LIBS} ${IE_SHARED_LIBS})

# MAKE Application
set(SOURCES application/main.cpp application/core_executor.cpp)
add_executable(NCSRecognition ${SOURCES})
if (IS_RASPBERRY)
    target_link_libraries(NCSRecognition NCSLib ${OpenCV_LIBS} ${WIRING_PI_LIB} ${IE_SHARED_LIBS} "stdc++fs")
else ()
    target_link_libraries(NCSRecognition NCSLib ${OpenCV_LIBS} ${IE_SHARED_LIBS} "stdc++fs")
endif ()

# MAKE Install
install(TARGETS NCSLib NCSRecognition
        DESTINATION ${PROJECT_SOURCE_DIR}/install/bin)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
        DESTINATION ${PROJECT_SOURCE_DIR}/install)
