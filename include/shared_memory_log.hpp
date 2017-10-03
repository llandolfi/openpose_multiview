#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>

struct trace_queue
{
   enum {image_size = 640*480*3 * sizeof(unsigned char)};
   enum {depth_size = 640*480*2 * sizeof(unsigned char)};

   trace_queue()
      :  message_in(false)
   {}

   //Mutex to protect access to the queue
   boost::interprocess::interprocess_mutex      mutex;

   //Condition to wait when the queue is empty
   boost::interprocess::interprocess_condition  cond_empty;

   //Condition to wait when the queue is full
   boost::interprocess::interprocess_condition  cond_full;

   //Put here the payload
   unsigned char RGB[image_size];
   unsigned char depth[depth_size];

   //Is there any message
   bool message_in;
};