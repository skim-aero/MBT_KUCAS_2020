/********************************************************************************
 * @file   MBT_KUCAS.cpp							*
 * @date   29th JUN 2020							*
 * @author Sukkeun Samuel Kim(samkim96@pusan.ac.kr)				*
 * @brief  Software for the KUCAS Project 2020 flight tests, main Header	*
 *******************************************************************************/

#include "MBT_KUCAS.h"

int main( int argc, char **argv )
{
    cv::VideoCapture cap( argv[1] );			// Imput video with OpenCV
    cv::VideoWriter video( "MBT_RESULT.avi", cv::VideoWriter::fourcc( 'F', 'M', 'P', '4' ), 22.0, cv::Size( 1920, 1080 ) );

    /**************************Initial Setting for DNN**************************/

    // [DETECT] Load names of classes
    std::string classesFile = "KUCAS.names";
    std::ifstream ifs( classesFile.c_str() );
    std::string line;
    while ( getline( ifs, line ) ) classes.push_back( line );
    
    // [DETECT] Give the configuration and weight files for the model
    cv::String modelConfiguration = "yolov3-KUCAS.cfg";
    cv::String modelWeights = "yolov3-KUCAS.weights";

    // [DETECT] Load the network
    std::cout << "[DETECT] Loading Deep Learning Backend: CUDA" << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromDarknet( modelConfiguration, modelWeights );
    net.setPreferableBackend( cv::dnn::DNN_BACKEND_CUDA );
    net.setPreferableTarget( cv::dnn::DNN_TARGET_CUDA );

    /************************Initial Setting for VISP***************************/
    std::string opt_videoname = "KUCAS.mp4";
    std::string opt_modelname = "CTSW.cao";
    std::string parentname = vpIoTools::getParent( opt_modelname );
    std::string objectname = vpIoTools::getNameWE( opt_modelname );

    if ( !parentname.empty() ) objectname = parentname + "/" + objectname;

    std::cout << "Video name: " << opt_videoname << std::endl;
    std::cout << "Tracker requested config files: " << objectname << ".[init, "
              << "xml, cao]" << std::endl;

    vpImage<unsigned char> I;				// Image
    vpCameraParameters cam;
    vpHomogeneousMatrix cMo;				// cMo

    cap >> src;
    PreProc( src, src_hist, 15, 45 );
    vpImageConvert::convert( src, I );

    vpDisplay *display = NULL;
    display = new vpDisplayOpenCV;
    display->init( I, 100, 100, "MBT_KUCAS" );

    vpMbGenericTracker tracker;				// vpMbGenericTracker class define

    tracker.loadModel( objectname + ".cao" );		// Load cao file
    //tracker.loadConfigFile( objectname + ".xml" );	// Load xml file
    tracker.loadConfigFile( "KUCAS.xml" );	// Load xml file

    tracker.initClick( I, objectname + ".init", true );	// Initial Point Click at here
    //tracker.initFromPoints( I, objectname + ".init");	
    std::ofstream fp( "results.txt" );			// Text out

    /**************************Initial Setting for VISP*************************/

    int right;						// Bounding box right line
    int bottom;						// Bounding box bottom line
    int n = 0;

    double ts, te = 0;

    while ( cap.isOpened() )
    {
	ts = cv::getTickCount();

	cap >> src;

	if ( src.empty() ) break;

	// Cloning src to src_dnn 
	cv::Mat src_dnn = src.clone();

	// Preprocessing for Harris corner, binarization/thresholding/morphological process, defined in VidProc.cpp
	PreProc( src, src_hist, 15, 45 );	// 15, 45 (Previous one), 15, 30 for new case

	// Deep Learing based object detection, dfined in DetecObj.cpp
	dnndetect( net, src_dnn, blob, classes );

	right = left + width;
	bottom = top + height;

	// Harris corner function defined in VidProc.cpp
	HarrisCo( 0, 0, src_hist );
	cv::cvtColor( src_hist, src_bgr, cv::COLOR_GRAY2BGR ); //Grey to BGR for video write

	// Draw boinding box and prediction, defined in DetecObj.cpp
	drawPred( 0, conf_main, left, top, right, bottom, src_bgr, classes );

	te = cv::getTickCount();

	UpdPoint( ts, te, src_bgr );
	vpImageConvert::convert( src_bgr, I );
	tracker.initFromPoints( I, "test.init" );	
	vpDisplay::display( I );

	tracker.getPose( cMo );				// Get Pose
	for ( int i = 0; i < 16; ++i )
	{
	    fp << cMo.data[i] << "  ";
	}
	fp << "\n";

	tracker.getCameraParameters( cam );

	// Display
	tracker.display( I, cMo, cam, vpColor::green, 2, true );
	vpDisplay::flush( I );

	++n;

	if ( vpDisplay::getClick( I, false ) || cv::waitKey( 1 ) == 27 ) break;
    }

    vpDisplay::getClick( I );
    cap.release();

    return 0;

//! [Cleanup]
    vpXmlParser::cleanup();

}
