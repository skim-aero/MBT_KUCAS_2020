/********************************************************************************
 * @file   DetecObj.h								*
 * @date   23rd JUN 2020							*
 * @author Sukkeun Samuel Kim(samkim96@pusan.ac.kr)				*
 * @brief  Software for the KUCAS Project 2020 flight tests, Detect Object	*
 *******************************************************************************/

#ifndef KUCAS_DetecObj_H_
#define KUCAS_DetecObj_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define CONFTHRES 	0.5									// confThreshold
#define NMSTHRES 	0.4									// nmsThreshold
#define INPW 		416									// inpWidth
#define INPH 		416									// inpHeight
#define PI 		3.14159265								// PI

extern int width, height, left, top, thresh;
extern float conf_main;
extern cv::Mat src_hist, src;

void dnndetect(cv::dnn::Net& net, cv::Mat& frame, cv::Mat& blob, std::vector<std::string> classes);

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out, std::vector<std::string> classes);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string> classes);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);

#endif  // KUCAS_DetecObj_H_
