#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <chrono>

struct Payload
{  
   enum {max_size = (480*640) * sizeof(unsigned char)};

   Payload(int w, int h)
      :  width_(w), height_(h) {}

   uint getRGBSize()
   {
      return width_ * height_ * 3;
   }

   uint getDepthSize()
   {
      return width_ * height_ * 2;
   }



   //Put here the payload
   unsigned char RGB[max_size * 3];
   unsigned char depth[max_size * 2];


   int width_;
   int height_;

   std::chrono::time_point<std::chrono::high_resolution_clock> time_;
   uint frame_;
};
