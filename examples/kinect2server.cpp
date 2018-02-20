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

  if(argc > 1)
  {
    freenectprocessor = static_cast<Processor>(atoi(argv[1]));
  }
      
  K2G k2g(freenectprocessor);

  cv::Mat color, depth;

  while(!to_stop){

    k2g.get(color, depth);
    // Showing only color since depth is float and needs conversion
    cv::namedWindow("color", CV_WINDOW_AUTOSIZE);
    cv::imshow("color", color);
    int c = cv::waitKey(1);

    if (c == 27)
    {
      exit(-1);
    }

  }

  k2g.shutDown();
  return 0;
}

