/********************************************************************************
 * @file   MBT_KUCAS.h								*
 * @date   25th JUN 2020							*
 * @author Sukkeun Samuel Kim(samkim96@pusan.ac.kr)				*
 * @brief  Software for the KUCAS Project 2020 flight tests, main Header	*
 *******************************************************************************/

#ifndef KUCAS_MBT_KUCAS_H_
#define KUCAS_MBT_KUCAS_H_

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <visp3/core/vpIoTools.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/mbt/vpMbGenericTracker.h>
#include <visp3/io/vpVideoReader.h>
#include <visp3/io/vpVideoWriter.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "VidProc.h"
#include "DetecObj.h"

std::vector<std::string> classes;
cv::Mat src, src_gray, src_hist, src_bgr, blob;

int thresh = 100;
int max_thresh = 255;
int width, height, left, top;
int corner[5][2];
float conf_main;

#endif  // KUCAS_MBT_KUCAS_H_

