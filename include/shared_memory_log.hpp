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

   std::chrono::milliseconds time_;
   uint frame_;
};

struct K2Payload
{  

   uint getRGBSize()
   {
      return 1920 * 1080 * 4;
   }

   uint getDepthSize()
   {
      return 512 * 424 * 2;
   }



   //Put here the payload
   unsigned char RGB[1920 * 1080 * 4];
   unsigned char depth[424 * 512 * 2];

   int color_width_ = 1920;
   int color_height_ = 1080;

   int depth_width = 512;
   int depth_height = 424;

   std::chrono::milliseconds time_;
   uint frame_;
};

