#include "kinect1/freenect_grabber.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "ipcpooledchannel.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "shared_memory_log.hpp"


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

  	shared_memory_object shm (open_or_create, "kinect1", read_write);

  	shm.truncate(sizeof(trace_queue));

  	mapped_region region(shm, read_write);

  	void* message_addr = (void*)region.get_address();

  	trace_queue * data = new(message_addr) trace_queue;

	while (!to_stop)
  	{

  		kserver.postRGBD(RGB, depth);

  		if(!RGB.empty())
  		{

  
  			if(count % 2 == 0)
  			{

  				//TODO: simply acquire the log
  				scoped_lock<interprocess_mutex> lock(data->mutex);

  				memcpy(data->RGB, RGB.data, (640*480*3)*sizeof(uint8_t));
  				memcpy(data->depth, depth.data, (640*480)*sizeof(uint16_t));

  				//Notify that there is someting to read
  				data->message_in = true;
  				data->cond_empty.notify_one();

			}
		}

		count ++;
  	}

  return 0;
}