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

/*g++ ./src/stereocam.cpp  -lopenpose -DUSE_CAFFE -lopencv_core -lopencv_highgui -I /usr/local/cuda-8.0/include/ -L /usr/local/cuda-8.0/lib64  -lcudart -lcublas -lcurand -L /home/lando/projects/openpose_stereo/openpose/3rdparty/caffe/distribute/lib/  -I /home/lando/projects/openpose_stereo/openpose/3rdparty/caffe/distribute/include/ -lcaffe -DUSE_CUDNN  -std=c++11 -pthread -fPIC -fopenmp -O3 -lcudnn -lglog -lgflags -lboost_system -lboost_filesystem -lm -lboost_thread -luvc  -o prova.a
*/
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging


DEFINE_string(video,                    "../data/example/video.avi",             "Full path of the video file");

DEFINE_bool(verify,                     false,                                   "Show projection of triangulated points"); 

DEFINE_string(file,                     "../data/example/keypoints.csv",          "Full path of the CSV file with 2D points");                 

bool keep_on = true;


int main(int argc, char **argv) {

  // Parsing command line flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /*use a videocapture and get keypoints from it. Assumes the video is a stereo video*/
  cv::VideoCapture cap(FLAGS_video);
  if( !cap.isOpened())
  {
    std::cout << "Could not read video file. Exiting." << std::endl;
    return -1;
  }

  //TODO get resolution from videocapture and 
  cv::Size S = cv::Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

  std::string res = std::to_string(S.width/2) + "x" + std::to_string(S.height);

  PoseExtractor *stereoextractor = new PoseExtractorFromFile(argc, argv, res, FLAGS_file);
    
  stereoextractor->init();

  cv::Mat image;
  double error = 0.0;

  while(keep_on)
  {

    cv::Mat pnts3D;
    cap >> image;

    double error = stereoextractor->go(image,FLAGS_verify,pnts3D,&keep_on);

  }

  stereoextractor->destroy();

  return 0;
}