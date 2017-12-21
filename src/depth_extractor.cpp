#include "stereoprocess.h"
#include "opencv2/cudastereo.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>

DEFINE_string(kernel_output,              "",      "Path of the kernel output file");

std::map<int, std::string> body_map = {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Background"},
};


bool inited = false;


DepthExtractor::DepthExtractor(int argc, char **argv, DepthCamera & camera, const std::string & depth_video) : PoseExtractor(argc, argv, camera)
{

  cam_ = camera;
  depthpath_ = depth_video;
  kernel_output_ = FLAGS_kernel_output;

  if(kernel_output_ != "")
  {
    kernelcsv_.open(kernel_output_);
  }

}

std::string DepthExtractor::pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time)
{

  Json::Value points;
  Json::Value colors;


  for(int i = 0; i < pnts.cols; i++)
  { 
    cv::Vec3d point = pnts.at<cv::Vec3d>(0,i);
    Json::Value jpoint;
    jpoint["x"] = point[0];
    jpoint["y"] = -point[1];
    jpoint["z"] = point[2];
    points.append(jpoint);
  }

  
  colors["r"] = 1.0;
  colors["g"] = 0;
  colors["b"] = 0;
  

  Json::Value root;
  root["type"] = "bodypoints";
  root["frame"] = frame;
  root["id"] = "uniquestring";
  root["radius"] = 0.08;
  root["pointorder"] = "openpose";
  root["color"] = colors;
  root["points"] = points;
  root["timestamp"] = time;

  Json::FastWriter writer;
  Json::StyledWriter writerp;

  //std::cout << "sending: " << std::endl;
  //std::cout << writerp.write(root) << std::endl;

  return writer.write(root);
}

/*
* x: point coordinate in pixel
* y: point coordinate in pixel
* d: disparity at point (x,y)
*/
cv::Point3d DepthExtractor::getPointFromDepth(double u, double v, double z)
{

  if(z == 0.0)
  {
    return cv::Point3d(0,0,0);
  }

  double fx = cam_.intrinsics_.at<double>(0,0);
  double fy = cam_.intrinsics_.at<double>(1,1);
  double cx = cam_.intrinsics_.at<double>(0,2);
  double cy = cam_.intrinsics_.at<double>(1,2);

  double Z = z;
  double X = ((v - cx) * Z)/fx;
  double Y = ((u - cy) * Z)/fy;

  return cv::Point3d(X,Y,Z);

}

double DepthExtractor::getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D)
{
  if(pnts3D.empty())
  {
    return 0.0;
  }

  cv::Mat points2D; 

  cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_.intrinsics_,cv::Vec4d(0,0,0,0),points2D);

  cv::transpose(points2D,points2D);

  return cv::norm(points2D - cam0pnts);
}

void DepthExtractor::kernel2CSV(int idx, const cv::Mat & kernel)
{
  //TODO: get current frame, get body part from i, write kernel left to right up to bottom
  std::vector<int> depths;
  int frame = cur_frame_;
  std::string bodypart = body_map[idx];

  for(int i=0; i < kernel.rows; i++)
  {
    for(int j=0; j < kernel.cols; j++)
    {
      depths.push_back((int)kernel.at<uint16_t>(i,j));
    }
  }

  kernelcsv_ << cur_frame_ << " " << bodypart << " ";

  for(auto &a : depths)
  {
    kernelcsv_ << a << " ";
  }

  kernelcsv_ << "\n";
}

double DepthExtractor::triangulate(cv::Mat & finalpoints)
{ 

  double epsilon = 100;
  cv::Mat cam0pnts;
  double confidence = 0.0;
  opArray2Mat(poseKeypointsL_, cam0pnts);


  if(cam0pnts.empty())
  {
    return 0.0;
  }

  std::vector<cv::Point3d> points3D;
  std::vector<cv::Point2d> points2D;

   
  //filterVisible(cam0pnts, cam0pnts);

  //Maybe smooth a little bit like in disparity?
  for( int i = 0; i < cam0pnts.cols; i++)
  { 
    cv::Point3d pwithnot = cam0pnts.at<cv::Point3d>(0,i);
    confidence = pwithnot.z;
    cv::Point2d keypoint(cvRound(pwithnot.x), cvRound(pwithnot.y));

   // std::cout << "Keypoints " << poseKeypointsL_.toString() << std::endl;
   // std::cout << "rows: " << depth_.rows << " columns: " << depth_.cols << std::endl;

    if(keypoint.x > depth_.cols)
    {
      std::cout << "Attenzione x " << keypoint.x << std::endl;
      exit(-1);
    }

    if(keypoint.y > depth_.rows)
    {
      std::cout << "Attenzione y " << keypoint.y << std::endl;
      exit(-1);
    }

    cv::Mat kernel;
    cv::Point3d point = getPointFromDepth(keypoint.y,keypoint.x,
                        //(double)depth_.at<uint16_t>(cvRound(keypoint.y), cvRound(keypoint.x)));
                        Pool(depth_, keypoint.y, keypoint.x, 7, gaussianAvg, kernel));

    //uint16_t ddepth = depth_.at<uint16_t>(keypoint.y, keypoint.x);

    if(FLAGS_kernel_output != "")
    {
      kernel2CSV(i,kernel);
    }

    if(point.x != 0.0 && point.y != 0.0 && point.z != 0.0 && confidence > 0.2)
    {
      point = point / 1000;
      points3D.push_back(point);
      points2D.push_back(keypoint);
    }
    else
    { 
      double n = std::numeric_limits<double>::quiet_NaN();
      points3D.push_back(cv::Point3d(n,n,n));
      points2D.push_back(keypoint);
    }
  }

  //TODO: remove zeros also from points3D and the correspondent from points 2D
  cam0pnts = cv::Mat(points2D);
  cv::transpose(cam0pnts, cam0pnts);

  //std::cout << "Nose: " << points3D[0] << std::endl;

  cv::Mat tmp = cv::Mat(points3D);
  finalpoints = tmp.clone();
  cv::transpose(finalpoints, finalpoints);

  double error = getRMS(cam0pnts, finalpoints);

  return error;
}
/**
* Converts depth into RGB image for MPEG encoding
*/
void encodeDepth(const cv::Mat & depth, cv::Mat & output)
{
  cv::Mat dc1 = cv::Mat(depth.rows, depth.cols, CV_8UC1);
  cv::Mat dc2 = cv::Mat(depth.rows, depth.cols, CV_8UC1);
  cv::Mat dummy = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);

  for (int i=0; i < depth.rows; i++)
  {
    for(int j=0; j < depth.cols; j++)
    {
      uint16_t val = depth.at<uint16_t>(i,j);
      uint8_t * ptr = (uint8_t*)&val;

      dc1.at<uint8_t>(i,j) = ptr[0];
      dc2.at<uint8_t>(i,j) = ptr[1];

    }
  }

  std::vector<cv::Mat> channels;
  channels.push_back(dc1);
  channels.push_back(dc2);
  channels.push_back(dummy);

  cv::merge(channels, output);
}

/**
* Decodes an encoded depth frame from kinect
*/
void decodeDepth(const cv::Mat & rgb, cv::Mat & depth)
{

  depth = cv::Mat(rgb.rows, rgb.cols, CV_16UC1);
  uint8_t buf[2];
  uint16_t * decodecptr = (uint16_t*)&buf;

  for (int i=0; i < depth.rows; i++)
  {
    for(int j=0; j < depth.cols; j++)
    {
      cv::Vec3b value = rgb.at<cv::Vec3b>(i,j);
      buf[0] = value[0];
      buf[1] = value[1];
      depth.at<uint16_t>(i,j) = *decodecptr;
    }
  }
}

void DepthExtractor::appendFrame(const ImageFrame & myframe)
{

 outputVideo_ << myframe.color_;
 
 //convert depth in normal codec: from single channel 16 bit 
 cv::Mat depthtosave;
 encodeDepth(myframe.depth_, depthtosave);
 depthoutput_ << depthtosave;  

 timefile_ << std::to_string(myframe.time_stamp_.count()) << "\n";
}


void DepthExtractor::prepareOutputVideo(const std::string & path)
{
  //TODO: parse resolution from instance fields
  cv::Size S = cv::Size(640, 480);
  poseVideo_.open(path, CV_FOURCC('D','I','V','X'), 10, S, true);
  if (!poseVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }
}

void DepthExtractor::prepareVideo(const std::string & path)
{

  cv::Size S = cv::Size(640, 480);
  outputVideo_.open(path, CV_FOURCC('D','I','V','X'), 30, S, true);
  if (!outputVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }
  std::string depthpath = "";

  if (depthpath_ == "")
  {
    depthpath = path + "depth.avi";
  }
  else
  {
    depthpath = depthpath_;
  }

  depthoutput_.open(depthpath, CV_FOURCC('D','I','V','X'), 30, S, true);

  if (!depthoutput_.isOpened())
  {
      std::cout  << "Could not open the depth output video for write: " << std::endl;
      exit(-1);
  }

  timefile_.open(path + "_timestamps.txt");
  jsonfile_.open(path + ".json"); 
}

void DepthExtractor::process(const std::string & write_keypoint, bool viz)
{ 

  PoseProcess(pose_params_, RGB_, poseKeypointsL_, outputImageL_);

  if( write_keypoint != "")
  {
    emitCSV(outputfile_, poseKeypointsL_, 0, cur_frame_);
  }

  if( viz)
  {
    visualize(&keep_on);
  }
}

void DepthExtractor::extract(const ImageFrame & m)
{ 

  RGB_ = m.color_;
  depth_ = m.depth_; 

  if(!live_)
  {
    decodeDepth(depth_, depth_);
  }

  cur_frame_ = cur_frame_ + skip_ + 1;
}


void DepthExtractor::visualize(bool* keep_on)
{

  cv::namedWindow("Keypoints", CV_WINDOW_AUTOSIZE);
  cv::imshow("Keypoints", outputImageL_);

  short k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}

void DepthExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 

  cv::Mat verification = outputImageL_.clone();

  if(!pnts.empty())
  {

    std::vector<cv::Point2d> points2D(pnts.cols);

    cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_.intrinsics_,cv::Vec4d(0,0,0,0),points2D);

    for (auto & c : points2D)
    { 
      cv::circle(verification,c,5,cv::Scalar(255,0,0),5);
    }

  }

  if(videooutput_)
  {
    poseVideo_ << verification;
  }

  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  cv::imshow("Verification", verification);
  
  short k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
  if (k == 's')
  {
    cv:imwrite("../data/3Dpoints.jpg", verification);
  }
}