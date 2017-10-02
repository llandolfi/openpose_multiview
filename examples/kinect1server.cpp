#include "kinect1/freenect_grabber.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>


const int distance = 7000;
bool to_stop = false;
uint64_t count = 0;


int main( int argc, char** argv )
{

	std::cout << "Ciao sono k1 server" << std::endl;

  	//TODO: this executable should read from Kinect 1 using freenect_grabber and post the frames in an interprocess pooled channel 
  	freenectServer kserver;
  	cv::Mat RGB,depth;

	while (!to_stop)
  	{

  		kserver.postRGBD(RGB, depth);

  		if(!RGB.empty())
  		{	

  			if(count % 2 == 0)
  			{
  				std::cout << "Case 1" << std::endl;
  
				cv::namedWindow("RGB", CV_WINDOW_AUTOSIZE);
  				cv::imshow("RGB", RGB);

  				double min,max;

  				cv::minMaxIdx(depth, &min, &max);
  				cv::Mat adjdepth;
  				cv::convertScaleAbs(depth, adjdepth, 255/ max);

  				cv::namedWindow("Depth", CV_WINDOW_AUTOSIZE);
  				cv::imshow("Depth", adjdepth);

  	
  				int k = cvWaitKey(0);
  				if (k == 27)
  				{
  	    			to_stop = true;
  				}
			}
			else
			{	
				std::cout << "Case 2" << std::endl;
			}

			std::cout << "Reading RGB" << std::endl;
			std::cout << RGB.cols  << " " << RGB.rows << std::endl;
			//std::cout << depth.cols << " " << depth.rows << std::endl;

			std::cout << "Reading depth" << std::endl;
			std::cout << depth.cols << " " << depth.rows << std::endl;

		}

		count ++;
  	}

  return 0;
}