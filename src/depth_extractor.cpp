#include "stereoprocess.h"
#include "opencv2/cudastereo.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>

bool inited = false;


DepthExtractor::DepthExtractor(int argc, char **argv, const std::string resolution) : PoseExtractor(argc, argv, resolution)
{
  std::cout << "resolution: " << resolution << std::endl;
  pcam_ = new DepthCamera();
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

  double fx = pcam_->intrinsics_.at<double>(0,0);
  double fy = pcam_->intrinsics_.at<double>(1,1);
  double cx = pcam_->intrinsics_.at<double>(0,2);
  double cy = pcam_->intrinsics_.at<double>(1,2);

  double Z = z;
  double X = ((u - cx) * Z)/fx;
  double Y = ((v - cy) * Z)/fy;

  return cv::Point3d(X,Y,Z);

}

double DepthExtractor::getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D)
{
  if(pnts3D.empty())
  {
    return 0.0;
  }

  cv::Mat points2D; 

  cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),pcam_->intrinsics_,cv::Vec4d(0,0,0,0),points2D);

  cv::transpose(points2D,points2D);

  return cv::norm(points2D - cam0pnts);
}

double DepthExtractor::triangulate(cv::Mat & finalpoints)
{ 

  double epsilon = 100;
  //I can take all the points negleting if they belong to a specific person 
  //how can I know if the points belong to the same person? 
  cv::Mat cam0pnts;
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

    cv::Point3d kpwc = cam0pnts.at<cv::Point3d>(0,i);
    cv::Point2d keypoint(cvRound(kpwc.x), cvRound(kpwc.y));

    cv::Point3d point = getPointFromDepth(keypoint.y,keypoint.x,
                        (double)depth_.at<uint16_t>(cvRound(keypoint.y), cvRound(keypoint.x)));
                        //Pool(depth_, keypoint.x, keypoint.y, 1,MinPool));

    uint16_t ddepth = depth_.at<uint16_t>(cvRound(keypoint.y), cvRound(keypoint.x));

    if(ddepth > 0 && point.x != 0.0 && point.y != 0.0)
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

void DepthExtractor::prepareVideo(const std::string & path)
{

  cv::Size S = cv::Size(640, 480);
  outputVideo_.open(path, CV_FOURCC('P','I','M','1'), 30, S, true);
  if (!outputVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }

  std::string depthpath = path + "depth.avi";
  depthoutput_.open(depthpath, CV_FOURCC('P','I','M','1'), 30, S, true);
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

  cur_frame_ ++;
  RGB_ = m.color_;

  if(live_)
  {

  depth_ = m.depth_;
  }

  else
  {

    if(!inited)
    {
      //TODO: set up a videocapture for the depth video
      depthcap_ = cv::VideoCapture(videoname_ + "depth.avi");
      if( !depthcap_.isOpened())
      {
        std::cout << "Could not read depth video file. Exiting." << std::endl;
        exit(-1);
      }
    }

    cv::Mat tmpdepth;
    depthcap_ >> tmpdepth;

    decodeDepth(tmpdepth, depth_);
  }

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

    cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),pcam_->intrinsics_,cv::Vec4d(0,0,0,0),points2D);

    for (auto & c : points2D)
    { 
      cv::circle(verification,c,5,cv::Scalar(255,0,0),5);
    }

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