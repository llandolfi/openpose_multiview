#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>

struct trace_queue
{
   enum {max_size = 1920*1080*3 * sizeof(unsigned char)};

   trace_queue(int w, int h)
      :  message_in(false), width_(w), height_(h)
   {}

   uint getRGBSize()
   {
      return width_ * height_ * 3;
   }

   uint getDepthSize()
   {
      return width_ * height_ * 2;
   }

   //Mutex to protect access to the queue
   boost::interprocess::interprocess_mutex      mutex;

   //Condition to wait when the queue is empty
   boost::interprocess::interprocess_condition  cond_empty;

   //Condition to wait when the queue is full
   boost::interprocess::interprocess_condition  cond_full;

   //Put here the payload
   unsigned char RGB[max_size];
   unsigned char depth[max_size];

   //Is there any message
   bool message_in;

   int width_;
   int height_;
};