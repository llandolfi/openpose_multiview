#include "k2g.h"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "shared_memory_log.hpp"
#include "ipcpooledchannel.hpp"
#include <chrono>


bool to_stop = false;
uint64_t count = 0;

void wait()
{
std::cout << "press a key to terminate " << std::endl;
 std::cin.get();
 to_stop = true;
}


int main(int argc, char * argv[])
{ 

  Processor freenectprocessor = OPENGL;
  uint cur_frame = 0;

  if(argc > 1)
  {
    freenectprocessor = static_cast<Processor>(atoi(argv[1]));
  }
      
  K2G k2g(freenectprocessor);

  cv::Mat color, depth, ir;

  IPCPooledChannel<K2Payload> pc("kinect2",WriterTag(),1,DiscardPolicy::DiscardOld,argc > 1);


  while(!to_stop){

    k2g.get(color, depth);

    cv::Mat depth_16(cv::Size(depth.rows,depth.cols),CV_16U);

    std::cout << "color.rows " << color.rows << " color.cols " << color.cols << std::endl;
    std::cout << "depth.rows " << depth.rows << " depth.cols " << depth.cols << " depth.channels " << depth.channels() <<std::endl;

  if(!color.empty())
    {

      K2Payload * data = new K2Payload();

      data = pc.writerGet();

      cur_frame ++;

      depth.convertTo(depth_16, CV_16U);

      memcpy(data->RGB, color.data, (1920*1080*4)*sizeof(uint8_t));
      memcpy(data->depth, depth_16.data, (512*424)*sizeof(uint16_t));

      data->frame_ = cur_frame;

      data->time_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

      pc.writerDone(data);
    
    }


    // Showing only color since depth is float and needs conversion
    /*cv::namedWindow("depth", CV_WINDOW_AUTOSIZE);
    cv::imshow("depth", depth_16); */
    int c = cv::waitKey(2);

    if (c == 27)
    {
      to_stop = true;
    }

  }

  pc.remove();
  k2g.shutDown();
  return 0;
}

