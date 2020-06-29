add_library(DetecObj DetecObj.cpp)

find_package(OpenCV)

target_link_libraries(DetecObj ${OpenCV_LIBS})
