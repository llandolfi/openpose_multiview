#include "kinect1/freenect_grabber.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "ipcpooledchannel.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>


const int distance = 7000;
bool to_stop = false;
uint64_t count = 0;



int main( int argc, char** argv )
{
	using namespace boost::interprocess;

	std::cout << "Ciao sono k1 server" << std::endl;

	struct shm_remove
	 {
	    shm_remove() { shared_memory_object::remove("kinect1"); }
	    ~shm_remove(){ shared_memory_object::remove("kinect1"); }
	 } remover;

  	//TODO: this executable should read from Kinect 1 using freenect_grabber and post the frames in an interprocess pooled channel 
  	freenectServer kserver;
  	cv::Mat RGB,depth;

  	//TODO: at fist try with simple shared memory, then use or modify ipc pooled channel

  	shared_memory_object shm (create_only, "kinect1", read_write);

  	shm.truncate(sizeof(uint8_t)*640*480*3);

  	mapped_region region(shm, read_write);

  	std::memset(region.get_address(), 0, region.get_size());

  	void* message_addr = (void*)region.get_address();

	while (!to_stop)
  	{

  		kserver.postRGBD(RGB, depth);

  		if(!RGB.empty())
  		{

  
  			if(count % 2 == 0)
  			{


  				memcpy(message_addr, RGB.data, (640*480*3)*sizeof(uint8_t));

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
				if (k == 's')
				{
				  cv:imwrite("./depth.jpg", adjdepth);
				}
  
			}
		}

		count ++;
  	}

  return 0;
}