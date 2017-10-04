#include "libuvc/libuvc.h"
#include "stereoprocess.h"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdlib.h> 
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include "utilities.hpp"
#include <map>
#include <cassert>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "shared_memory_log.hpp"
#include "kinect1/freenect_grabber.hpp"

/*g++ ./src/stereocam.cpp  -lopenpose -DUSE_CAFFE -lopencv_core -lopencv_highgui -I /usr/local/cuda-8.0/include/ -L /usr/local/cuda-8.0/lib64  -lcudart -lcublas -lcurand -L /home/lando/projects/openpose_stereo/openpose/3rdparty/caffe/distribute/lib/  -I /home/lando/projects/openpose_stereo/openpose/3rdparty/caffe/distribute/include/ -lcaffe -DUSE_CUDNN  -std=c++11 -pthread -fPIC -fopenmp -O3 -lcudnn -lglog -lgflags -lboost_system -lboost_filesystem -lm -lboost_thread -luvc  -o prova.a
*/
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging


DEFINE_string(video,                    "",             "Use a video instead of the camera");

DEFINE_int32(fps,                       60,              "Camera capture speed. Frame per second");

DEFINE_bool(verify,                     false,            "Show projection of triangulated points"); 

DEFINE_bool(disparity,                  false,            "Use disparity map instead of triangulation to get 3D body points");  

DEFINE_bool(depth,                      false,            "Use depth map instead of triangulation to get 3D body points");  

DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");  

DEFINE_string(camera,                   "ZED",           "The camera used for streaming (ZED,K1)");  


PoseExtractor * stereoextractor;
bool keep_on = true;

std::map<std::string, int> camera_map = {{"ZED",0},{"K1",1}};

/* This callback function runs once per frame. Use it to perform any
 * quick processing you need, or have it put the frame into your application's
 * input queue. If this function takes too long, you'll start losing frames. */
void cb(uvc_frame_t *frame, void *ptr) {
  uvc_frame_t *bgr;
  uvc_error_t ret;

  /* We'll convert the image from YUV/JPEG to BGR, so allocate space */
  bgr = uvc_allocate_frame(frame->width * frame->height * 3);
  if (!bgr) {
    printf("unable to allocate bgr frame!");
    return;
  }

  /* Do the BGR conversion */
  ret = uvc_any2bgr(frame, bgr);
  if (ret) {
    uvc_perror(ret, "uvc_any2bgr");
    uvc_free_frame(bgr);
    return;
  }

  // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
  stereoextractor->init();

  IplImage* cvImg = cvCreateImageHeader(
        cvSize(bgr->width, bgr->height),
       IPL_DEPTH_8U,
       3);
     
  cvSetData(cvImg, bgr->data, bgr->width * 3); 

  cv::Mat pnts;
  cv::Mat image = cv::cvarrToMat(cvImg);

  double error = stereoextractor->go(image,FLAGS_verify,pnts,&keep_on);
  //std::cout << "error " << error << std::endl;
   
  cvReleaseImageHeader(&cvImg);
   
  uvc_free_frame(bgr);
}

int startZedStream()
{
  uvc_context_t *ctx;
  uvc_device_t *dev;
  uvc_device_handle_t *devh;
  uvc_stream_ctrl_t ctrl;
  uvc_error_t res;

  /* Initialize a UVC service context. Libuvc will set up its own libusb
   * context. Replace NULL with a libusb_context pointer to run libuvc
   * from an existing libusb context. */
  res = uvc_init(&ctx, NULL);

  if (res < 0) {
    uvc_perror(res, "uvc_init");
    return res;
  }

  puts("UVC initialized");

  /* Locates the first attached UVC device, stores in dev */
  res = uvc_find_device(
      ctx, &dev,
      0, 0, NULL); /* filter devices: vendor_id, product_id, "serial_num" */

  if (res < 0) {
    uvc_perror(res, "uvc_find_device"); /* no devices found */
  } else {
    puts("Device found");

    /* Try to open the device: requires exclusive access */
    res = uvc_open(dev, &devh);

    if (res < 0) {
      uvc_perror(res, "uvc_open"); /* unable to open device */
    } else {
      puts("Device opened");

      /* Print out a message containing all the information that libuvc
       * knows about the device */
      //uvc_print_diag(devh, stderr);

      /* Try to negotiate a 640x480 30 fps YUYV stream profile */
      res = uvc_get_stream_ctrl_format_size(
          devh, &ctrl, /* result stored in ctrl */
          UVC_FRAME_FORMAT_YUYV, /* YUV 422, aka YUV 4:2:2. try _COMPRESSED */
          //1344, 376, 100
          getWidth(FLAGS_resolution) * 2, getHeight(FLAGS_resolution), FLAGS_fps /* width, height, fps */
      );

      /* Print out the result */
      uvc_print_stream_ctrl(&ctrl, stderr);

      if (res < 0) {
        uvc_perror(res, "get_mode"); /* device doesn't provide a matching stream */
      } else {  
        /* Start the video stream. The library will call user function cb:
         *   cb(frame, (void*) 12345)
         */
        res = uvc_start_streaming(devh, &ctrl, cb, (void*)123450, 0);

        if (res < 0) {
          uvc_perror(res, "start_streaming"); /* unable to start stream */
        } else {
          puts("Streaming...");

          uvc_set_ae_mode(devh, 1); /* e.g., turn on auto exposure */

          /*wait for an environment variable to be set */
          //sleep(10); /* stream for 10 seconds */
          while(keep_on)
          {
            sleep(1);
          }

          /* End the stream. Blocks until last callback is serviced */
          uvc_stop_streaming(devh);
          puts("Done streaming.");
          cv::destroyAllWindows();
        }
      }

      /* Release our handle on the device */
      //uvc_close(devh);
      puts("Device closed");
    }

    /* Release the device descriptor */
    puts("RELEASING");
    //uvc_unref_device(dev);
  }

  /* Close the UVC context. This closes and cleans up any existing device handles,
   * and it closes the libusb context if one was not provided. */
  // uvc_exit(ctx);
  puts("UVC exited");
}

void startK1Stream()
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

  stereoextractor->init();

  while (keep_on)
  {
   
    if(!data->message_in)
    {
      scoped_lock<interprocess_mutex> lock(data->mutex);
      data->cond_empty.wait(lock);
    }
    
    RGB.data = static_cast<uchar*>(data->RGB);
    depth.data = static_cast<uchar*>(data->depth);

    cv::Mat pnts;

    stereoextractor->setDepth(depth);
    double error = stereoextractor->go(RGB,FLAGS_verify,pnts,&keep_on);
  
  }
}

int main(int argc, char **argv) {

  // Parsing command line flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if(FLAGS_disparity == true)
  {
    stereoextractor = new DisparityExtractor(argc, argv, FLAGS_resolution);
  }
  else
  {
    if(FLAGS_depth == true)
    {
      stereoextractor = new DepthExtractor(argc, argv, FLAGS_resolution);
    }
    else
    {
      stereoextractor = new StereoPoseExtractor(argc, argv, FLAGS_resolution);
    }
  }

  if( FLAGS_video == "" )
  {

    switch(camera_map[FLAGS_camera])
    {
      case 0: 
              std::cout << "Streaming from ZED" << std::endl;
              startZedStream();

      case 1: 
              std::cout << "Streaming from Kinect 1" << std::endl;
              startK1Stream();
    }
  }

  else
  {
    /*use a videocapture and get keypoints from it. Assumes the video is a stereo video*/
    cv::VideoCapture cap(FLAGS_video);
    if( !cap.isOpened())
    {
      std::cout << "Could not read video file. Exiting." << std::endl;
      return -1;
    }
      
    stereoextractor->init();

    while(keep_on)
    {

      cv::Mat image,pnts;
      cap >> image;

      double error = stereoextractor->go(image,FLAGS_verify,pnts,&keep_on);
    }

  } 

  stereoextractor->destroy();

  return 0;
}