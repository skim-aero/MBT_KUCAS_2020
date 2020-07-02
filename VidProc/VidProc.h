/********************************************************************************
 * @file   VidProc.h								*
 * @date   29th JUN 2020							*
 * @author Sukkeun Samuel Kim(samkim96@pusan.ac.kr)				*
 * @brief  Software for the KUCAS Project 2020 flight tests, video processing	*
 *******************************************************************************/

#ifndef KUCAS_VidProc_H_
#define KUCAS_VidProc_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

extern int width, height, left, top, thresh;
extern cv::Mat src_hist, src;

extern int corner[5][2];

void PreProc( cv::Mat& mat_in, cv::Mat& mat_out, const double thres_l, const double thres_h );
void HarrisCo( int, void*, cv::Mat &input );
void UpdPoint( double ts, double te, cv::Mat &input );

#endif  // KUCAS_VidProc_H_
