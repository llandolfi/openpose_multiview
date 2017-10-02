#include "kinect1/freenect_grabber.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "ipcpooledchannel.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>



const int distance = 7000;
bool to_stop = false;


int main( int argc, char** argv )
{

  using namespace boost::interprocess;

	std::cout << "Ciao sono k1 client" << std::endl;

  //TODO: this executable should read from Kinect 1 using interprocess channel

  shared_memory_object shm (open_only, "kinect1", read_only);

  mapped_region region(shm, read_only);

  cv::Mat RGB(480,640,CV_8UC3);

	while (!to_stop)
  {

     
    RGB.data = static_cast<uchar*>(region.get_address());

    cv::namedWindow("RGB read", CV_WINDOW_AUTOSIZE);
  	cv::imshow("RGB read", RGB);

  	int k = cvWaitKey(0);
  	if (k == 27)
  	{
    		to_stop = true;
  	} 	
  }

  return 0;
}