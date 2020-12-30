/********************************************************************************
 * @file   VidProc.cpp								*
 * @date   2nd JUL 2020								*
 * @author Sukkeun Samuel Kim(samkim96@pusan.ac.kr)				*
 * @brief  Software for the KUCAS Project 2020 flight tests, Video Processing	*
 *******************************************************************************/

#include "VidProc.h"

void PreProc( cv::Mat& mat_in, cv::Mat& mat_out, const double thres_l, const double thres_h)
{
    cv::cvtColor( mat_in, mat_in, cv::COLOR_BGR2GRAY);
    cv::equalizeHist( mat_in, mat_in);    
    cv::adaptiveThreshold( mat_in, mat_in, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, thres_l, thres_h );
    cv::Mat mask = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2, 2 ), cv::Point( 1, 1 ) );
    cv::morphologyEx( mat_in, mat_out, cv::MorphTypes::MORPH_CLOSE, mask );
    //cv::cvtColor( mat_in, mat_in, cv::COLOR_GRAY2BG R); //Grey to BGR for video write
}

void HarrisCo( int, void*, cv::Mat &input )
{
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros( src.size(), CV_32FC1 );

    int blockSize = 2;						// 2
    int apertureSize = 3;					// 3
    double k = 0.08;						// 0.04(after 0.08)
    int right = left + width;
    int bottom = top + height;
    int l = 0;
    int l_n = 0;
    int l_y = 2;

    cv::cornerHarris( input, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT );
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    for( int j = top; j < bottom; ++j )
    {
	for( int i = left; i < right; ++i )
	{
	    if ( (int) dst_norm.at<float>(j,i) > thresh ) 
	    {
		++l;
		cv::circle( input, cv::Point( i, j ), 5, cv::Scalar(0), 2, 8, 0 );

	    }
	}
    }

    int **point = new int*[l];
    for ( int i = 0; i < l; ++i )
    {
	point[i] = new int[l_y];
    }

    for( int i = left; i < right; ++i )
    {
	for( int j = top; j < bottom; ++j )
	{
	    if ( (int) dst_norm.at<float>(j,i) > thresh ) 
	    {
		point[l_n][0] = i;
		point[l_n][1] = j;
		//std::cout << "Point: " << point[l_n][1] << ", " << point[l_n][0] << "\n" << std::endl;
		++l_n;
	    }
	}
    }

    corner[4][1] = ( left + right ) / 2;
    corner[4][0] = ( top + bottom ) / 2;

    corner[0][1] = point[0][0];
    for ( int i = 0; i < l; ++i )
    {
	if ( corner[0][1] < point[i][0] ) 
	{
	    corner[0][1] = point[i][0];
	    corner[0][0] = point[i][1];
	}
    }

    corner[2][1] = point[0][0];
    corner[2][0] = point[0][1];
    for ( int i = 0; i < l; ++i )
    {
	if ( corner[2][1] < point[i][0] && point[i][0] < corner[0][1] - 30 )
	{
	    corner[2][1] = point[i][0];
	    corner[2][0] = point[i][1];
	}
    }

    corner[1][1] = point[20][0];
    for ( int i = l - 1; i > 0; --i )
    {
	if ( corner[1][1] > point[i][0] ) 
	{
	    corner[1][1] = point[i][0];
	    corner[1][0] = point[i][1];
	}
    }

    corner[3][1] = point[20][0];
    corner[3][0] = point[20][1];
    for ( int i = l - 1; i > 0; --i )
    {
	if ( corner[3][1] > point[i][0] && point[i][0] > corner[1][1] + 30)
	{
	    corner[3][1] = point[i][0];
	    corner[3][0] = point[i][1];
	}
    }

    for ( int i = 0; i < l; ++i )
    {
	delete [] point[i];
    }
    delete [] point;
}

void UpdPoint( double ts, double te, cv::Mat &input )
{

    double time = ( ( te - ts ) / cv::getTickFrequency() ) * 1000;
    double fps = 1 / ( time / 1000 );

    std::string label = cv::format( "Inference time : %.2f ms (%.2f FPS)", time, fps );
    cv::putText( input, label, cv::Point( 10, 15 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );

    std::string label_1 = cv::format( "Point 1 with 2D coordinates: %d, %d", corner[0][1], corner[0][0] ); 
    std::string label_2 = cv::format( "Point 2 with 2D coordinates: %d, %d", corner[1][1], corner[1][0] ); 
    std::string label_3 = cv::format( "Point 3 with 2D coordinates: %d, %d", corner[2][1], corner[2][0] ); 
    std::string label_4 = cv::format( "Point 4 with 2D coordinates: %d, %d", corner[3][1], corner[3][0] ); 
    std::string label_5 = cv::format( "Point 5 with 2D coordinates: %d, %d", corner[4][1], corner[4][0] ); 

    cv::putText( input, label_1, cv::Point( 10,  35 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );
    cv::putText( input, label_2, cv::Point( 10,  55 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );
    cv::putText( input, label_3, cv::Point( 10,  75 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );
    cv::putText( input, label_4, cv::Point( 10,  95 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );
    cv::putText( input, label_5, cv::Point( 10, 115 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );

    std::ofstream ofs;
    ofs.open("test.init", std::ofstream::out | std::ofstream::trunc);
    ofs << "5" << "\n" << " 4.2650 " << "-1.1700 " << " 0.0750"
	       << "\n" << "-4.2650 " << "-1.1700 " << " 0.0750"
	       << "\n" << " 1.1900 " << "-4.5600 " << "-0.5250"
	       << "\n" << "-1.1900 " << "-4.5600 " << "-0.5250"
	       << "\n" << "      0 " << " 1.5650 " << "-0.7500";

    ofs << "\n";
    ofs << "5" << "\n" << corner[0][0] << " " << corner[0][1]
	       << "\n" << corner[1][0] << " " << corner[1][1]
	       << "\n" << corner[2][0] << " " << corner[2][1]
	       << "\n" << corner[3][0] << " " << corner[3][1]
	       << "\n" << corner[4][0] << " " << corner[4][1];
    ofs.close();
}
