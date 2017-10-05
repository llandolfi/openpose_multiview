#include "kinect1/freenect_grabber.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "shared_memory_log.hpp"



bool to_stop = false;


int main( int argc, char** argv )
{

  using namespace boost::interprocess;

	std::cout << "k1 client Ready" << std::endl;

  shared_memory_object shm (open_only, "kinect1", read_write);

  mapped_region region
         (shm                       //What to map
         ,read_write //Map it as read-write
         );

  //Get the address of the mapped region
  void * addr = region.get_address();

  //Obtain a pointer to the shared structure
  trace_queue * data = static_cast<trace_queue*>(addr);

  cv::Mat RGB(480,640,CV_8UC3);
  cv::Mat depth(480,640,CV_16UC1);

	while (!to_stop)
  {
   
    if(!data->message_in)
    {
      scoped_lock<interprocess_mutex> lock(data->mutex);
      data->cond_empty.wait(lock);
    }
    
    RGB.data = static_cast<uchar*>(data->RGB);
    depth.data = static_cast<uchar*>(data->depth);

    cv::namedWindow("RGB read", CV_WINDOW_AUTOSIZE);
  	cv::imshow("RGB read", RGB);

    double min,max;

    cv::minMaxIdx(depth, &min, &max);
    cv::Mat adjdepth;
    cv::convertScaleAbs(depth, adjdepth, 255/ max);

    cv::namedWindow("Depth", CV_WINDOW_AUTOSIZE);
    cv::imshow("Depth", adjdepth);

  	int k = cvWaitKey(10);
  	if (k == 27)
  	{
    		to_stop = true;
  	} 	
  }

  return 0;
}