project(KUCAS)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14) #If error occurs due to version upper than 4.0

set(CMAKE_CXX_STANDARD_REQUIRED ON) #If error occurs due to version upper than 4.0

find_package(VISP REQUIRED visp_core visp_mbt visp_io visp_gui visp_imgproc)
find_package(OpenCV REQUIRED)

# set the list of source files
set(MBT_cpp MBT_KUCAS.cpp)

configure_file(KUCAS.mp4 KUCAS.mp4 COPYONLY)
configure_file(KUCAS.cao KUCAS.cao COPYONLY)
configure_file(KUCAS.init KUCAS.init COPYONLY)
configure_file(KUCAS.xml KUCAS.xml COPYONLY)
configure_file(KUCAS.names KUCAS.names COPYONLY)
configure_file(yolov3-KUCAS.cfg yolov3-KUCAS.cfg COPYONLY)
configure_file(yolov3-KUCAS.weights yolov3-KUCAS.weights COPYONLY)

# Image Processing
include_directories(${PROJECT_SOURCE_DIR}/VidProc ${PROJECT_SOURCE_DIR}/DetecObj)

add_subdirectory(VidProc)
add_subdirectory(DetecObj)

set(MAIN_LIBS ${MAIN_LIBS} VidProc DetecObj)

add_executable(KUCAS MBT_KUCAS.cpp)
target_link_libraries(KUCAS ${MAIN_LIBS} ${VISP_LIBS} ${OpenCV_LIBS})



