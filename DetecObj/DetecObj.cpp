/********************************************************************************
 * @file   DetecObj.cpp								*
 * @date   23rd JUN 2020							*
 * @author Sukkeun Samuel Kim(samkim96@pusan.ac.kr)				*
 * @brief  Software for the KUCAS Project 2020 flight tests, Detect Object	*
 *******************************************************************************/

#include "DetecObj.h"

std::string label_t;

void dnndetect( cv::dnn::Net& net, cv::Mat& frame, cv::Mat& blob, std::vector<std::string> classes )
{
    // Create a 4D blob from a frame
    cv::dnn::blobFromImage( frame, blob, 1/255.0, cv::Size( INPW, INPW ), cv::Scalar( 0, 0, 0 ), true, false );

    // Sets the input to the network
    net.setInput( blob );

    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward( outs, getOutputsNames( net ) );

    // Remove the bounding boxes with low confidence
    postprocess( frame, outs, classes );

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile( layersTimes ) / freq;
    double fps = 1 / ( t / 1000 );

    label_t = cv::format( "Inference time : %.2f ms (%.2f FPS)", t, fps );
}

void postprocess( cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<std::string> classes )
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for ( size_t i = 0; i < outs.size(); ++i )
    {
	// Scan through all the bounding boxes output from the network and keep only the
	// ones with high confidence scores. Assign the box's class label as the class
	// with the highest score for the box.
	float* data = ( float* )outs[i].data;
	for ( int j = 0; j < outs[i].rows; ++j, data += outs[i].cols )
	{
	    cv::Mat scores = outs[i].row( j ).colRange( 5, outs[i].cols );
	    cv::Point classIdPoint;
	    double confidence;

	    // Get the value and location of the maximum score
	    minMaxLoc( scores, 0, &confidence, 0, &classIdPoint );
	    if ( confidence > CONFTHRES )
	    {
		int centerX = (int)( data[0] * frame.cols );
		int centerY = (int)( data[1] * frame.rows );

		width = (int)( data[2] * frame.cols );
		height = (int)( data[3] * frame.rows );
		left = centerX - width / 2;
		top = centerY - height / 2;

		classIds.push_back( classIdPoint.x );
		confidences.push_back( (float)confidence );
		boxes.push_back( cv::Rect( left, top, width, height ) );
	    }
	}
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes( boxes, confidences, CONFTHRES, NMSTHRES, indices );
    for ( size_t i = 0; i < indices.size(); ++i )
    {
	int idx = indices[i];
	cv::Rect box = boxes[idx];
	drawPred( classIds[idx], confidences[idx], box.x, box.y,
		  box.x + box.width, box.y + box.height, frame, classes );
	conf_main = confidences[idx];
    }
}

// Draw the predicted bounding box
void drawPred( int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string> classes )
{
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    //cv::putText( frame, label_t, cv::Point( 10, 15 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar( 0, 0, 255 ) );

    //Draw a rectangle displaying the bounding box
    cv::rectangle( frame, cv::Point( left, top ), cv::Point( right, bottom ), cv::Scalar( 255, 178, 50 ), 3 );

    //Get the label for the class name and its confidence
    std::string label = cv::format( "%.2f", conf );

    if ( !classes.empty() )
    {
	CV_Assert( classId < (int)classes.size() );
	label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;

    cv::Size labelSize = cv::getTextSize( label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine );
    top = std::max( top, labelSize.height );
    putText( frame, label, cv::Point( left, top - 10 ), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar( 0, 0, 0 ), 1 );
}

// Get the names of the output layers
std::vector<cv::String> getOutputsNames( const cv::dnn::Net& net )
{
    static std::vector<cv::String> names;

    if ( names.empty() )
    {
	//Get the indices of the output layers, i.e. the layers with unconnected outputs
	std::vector<int> outLayers = net.getUnconnectedOutLayers();

	//get the names of all the layers in the network
	std::vector<cv::String> layersNames = net.getLayerNames();

	// Get the names of the output layers in names
	names.resize( outLayers.size() );
	for ( size_t i = 0; i < outLayers.size(); ++i )
	    names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


