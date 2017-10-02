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
  				std::cout << "Reading " << std::endl;
  				std::cout << RGB.cols  << " " << RGB.rows << std::endl;
  				//std::cout << depth.cols << " " << depth.rows << std::endl;

  				cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  				cv::imshow("Verification", RGB);
  	
  				int k = cvWaitKey(2);
  				if (k == 27)
  				{
  	    			to_stop = true;
  				}
			}
		}

		count ++;
  	}

  return 0;
}