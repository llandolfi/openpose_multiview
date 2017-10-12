#include "stereoprocess.h"
#include "opencv2/cudastereo.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>



DepthExtractor::DepthExtractor(int argc, char **argv, const std::string resolution) : PoseExtractor(argc, argv, resolution)
{

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

  std::vector<cv::Point3d> points3D;
  std::vector<cv::Point2d> points2D;

   
  filterVisible(cam0pnts, cam0pnts);

  //Maybe smooth a little bit like in disparity?
  for( int i = 0; i < cam0pnts.cols; i++)
  { 
    cv::Point2d keypoint = cam0pnts.at<cv::Point2d>(0,i);

    cv::Point3d point = getPointFromDepth(keypoint.x,keypoint.y,
                        (double)depth_.at<uint16_t>(cvRound(keypoint.x), cvRound(keypoint.y)));
                        //Pool(depth_, keypoint.x, keypoint.y, 1,MinPool));

    double ddepth = depth_.at<uint16_t>(keypoint.x, keypoint.y);

    if(ddepth > 0)
    {
      point = point / 1000;
      points3D.push_back(point);
      points2D.push_back(keypoint);
    }
  }

  //TODO: remove zeros also from points3D and the correspondent from points 2D
  cam0pnts = cv::Mat(points2D);
  cv::transpose(cam0pnts, cam0pnts);

  //std::cout << "Nose: " << points3D[0] << std::endl;

  cv::Mat tmp = cv::Mat(points3D);
  finalpoints = tmp.clone();


  double error = getRMS(cam0pnts, finalpoints);

  return error;
}

void DepthExtractor::process(const std::string & write_video, const std::string & write_keypoint, bool viz)
{ 
  
  PoseProcess(pose_params_, RGB_, poseKeypointsL_, outputImageL_);

  if( write_video != "")
  { 
    outputVideo_ << RGB_;
  }

  if( write_keypoint != "")
  {
    emitCSV(outputfile_, poseKeypointsL_, 0, cur_frame_);
  }

  if( viz)
  {
    visualize(&keep_on);
  }
}

void DepthExtractor::extract(const cv::Mat & m)
{  
  cur_frame_ ++;
  RGB_ = m;
}

void DepthExtractor::visualize(bool* keep_on)
{

  cv::namedWindow("Keypoints", CV_WINDOW_AUTOSIZE);
  cv::imshow("Keypoints", outputImageL_);

  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}

void DepthExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 

  if(pnts.empty())
  {
    return;
  }

  std::vector<cv::Point2d> points2D(pnts.cols);

  cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),pcam_->intrinsics_,cv::Vec4d(0,0,0,0),points2D);

  cv::Mat verification = RGB_.clone();

  for (auto & c : points2D)
  { 
    cv::circle(verification,c,4,cv::Scalar(0,0,255),2);
  }


  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  cv::imshow("Verification", verification);
  
  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
  if (k == 's')
  {
    cv:imwrite("../data/3Dpoints.jpg", verification);
  }
}