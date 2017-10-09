#include "kinect1/freenect_grabber.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "shared_memory_log.hpp"
#include "ipcpooledchannel.hpp"
#include <chrono>

bool to_stop = false;
uint64_t count = 0;


int main( int argc, char** argv )
{
  using namespace boost::interprocess;

  std::cout << "Ciao sono k1 server" << std::endl;

    //TODO: this executable should read from Kinect 1 using freenect_grabber and post the frames in an interprocess pooled channel 
    freenectServer kserver;
    cv::Mat RGB,depth;

    int w = 640;
    int h = 480;

    uint cur_frame = 0;

    IPCPooledChannel<Payload> pc("kinect1",WriterTag(),3,DiscardPolicy::NoDiscard,argc > 1);

  while (!to_stop)
    {

      kserver.postRGBD(RGB, depth);

      if(!RGB.empty())
      {

  
        if(count % 2 == 0)
        {

          Payload * data = new Payload(640,480);

          data = pc.writerGet();
          
          data->width_ = w;
          data->height_ = h;
               
          cur_frame ++;

          memcpy(data->RGB, RGB.data, (w*h*3)*sizeof(uint8_t));
          memcpy(data->depth, depth.data, (w*h)*sizeof(uint16_t));
          data->frame_ = cur_frame;

          data->time_ = std::chrono::high_resolution_clock::now();

          pc.writerDone(data);

      }
    }

    count ++;
    }

    pc.remove();

  return 0;
}