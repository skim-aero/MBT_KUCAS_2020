add_library(VidProc VidProc.cpp)

find_package(OpenCV)

target_link_libraries(VidProc ${OpenCV_LIBS})
