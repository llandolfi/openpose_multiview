#include "stereoprocess.h"
#include "utilities.hpp"
#include <math.h>

DisparityExtractor::DisparityExtractor(int argc, char **argv, const std::string resolution) : StereoPoseExtractor(argc,argv,resolution){

  double f = cam_.intrinsics_left_.at<double>(0,0);
  double cx = cam_.intrinsics_left_.at<double>(0,2);
  double cy = cam_.intrinsics_left_.at<double>(1,2);
  double B = cam_.ST_[0];

  //TODO: build the Q matrix
  cv::Mat K4 = (cv::Mat_<double>(4,4) << f, 0.0, 0.0, cx, 0.0, f, 0.0, cy, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0);
  cv::Mat RT = cv::Mat::eye(4,4,CV_64FC1);
  RT.at<double>(0,3) = B;

  P_ = K4 * RT;
  iP_ = P_.inv();

    cv::Mat R1,R2,P1,P2;
    cv::Size img_size = cv::Size(cam_.width_,cam_.height_);

    cv::stereoRectify(cam_.intrinsics_left_,cam_.dist_left_,cam_.intrinsics_right_,cam_.dist_right_,img_size, cam_.SR_, cam_.ST_,
                      R1,R2,P1,P2,Q_, cv::CALIB_ZERO_DISPARITY, -1,img_size ,&roi1_, &roi2_);

}

DisparityExtractor::DisparityExtractor(int argc, char **argv, const std::string resolution, int fps) : StereoPoseExtractor(argc, argv, resolution, fps)
{

}

void DisparityExtractor::getDisparity()
{ 
  cv::Mat grayleft,grayright;

  cv::cvtColor(imageleft_,grayleft, CV_BGR2GRAY);
  cv::cvtColor(imageright_,grayright, CV_BGR2GRAY);

  gpuleft_.upload(grayleft);
  gpuright_.upload(grayright);


  disparter_->compute(gpuleft_,gpuright_,disparity_);
  //disparter_s_->compute(gpuleft_,gpuright_,disparity_);
}

/*
* x: point coordinate in pixel
* y: point coordinate in pixel
* d: disparity at point (x,y)
*/
cv::Point3d DisparityExtractor::getPointFromDisp(double u, double v, double d)
{

  double f = cam_.intrinsics_left_.at<double>(0,0);
  double cx = cam_.intrinsics_left_.at<double>(0,2);
  double cy = cam_.intrinsics_left_.at<double>(1,2);
  double b = -cam_.ST_[0];

  double Z = (f * b)/d;
  double X = ((u - cx) * Z)/f;
  double Y = ((v - cy) * Z)/f;

  return cv::Point3d(X,Y,Z);

}

double DisparityExtractor::avgDisp(const cv::Mat & disp, int u, int v, int side)
{

  double wlb,hlb;

  wlb = std::max(0,u - side);
  hlb = std::max(0,v - side);

  cv::Mat matrix(disp(cv::Rect(wlb,hlb,side,side)));

  double sum = cv::sum(matrix)[0];
  int nonzero = cv::countNonZero(matrix); 

  return sum / (double)nonzero;
}

double DisparityExtractor::maxDisp(const cv::Mat & disp, int u, int v, int side)
{ 

  double min,max,min_loc,max_loc = 0;

  double wlb,wub;
  double hlb,hub;

  wlb = std::max(0,u - side);
  hlb = std::max(0,v - side);

  wub = std::min(cam_.width_, u + side + 1);
  hub = std::min(cam_.height_,v + side + 1);

  cv::Mat matrix(disp(cv::Rect(wlb,hlb,side,side)));

  cv::minMaxLoc(matrix,&min,&max);

  return max;
}

double DisparityExtractor::triangulate(cv::Mat & output) 
{

  getDisparity();

  cv::Mat disp,disp8;
  cv::Mat cam0pnts,cam1pnts;
  cv::Mat xyz;

  disparity_.download(disp);  
  getPoints(cam0pnts,cam1pnts);
  filterVisible(cam0pnts,cam1pnts,cam0pnts,cam1pnts);

  disp.convertTo(disp8, CV_8U, 255/(disparter_->getNumDisparities()*16.));

  output = cv::Mat(1,cam0pnts.cols,CV_64FC3);

  for(int i=0; i < cam0pnts.cols; i++)
  {

    cv::Vec2d p = cam0pnts.at<cv::Vec2d>(0,i);
    //disparity at the considered pixel 
    double dispatpoint = (double)disp.at<uint16_t>(cvRound(p[0]),cvRound(p[1]));
    //disparity in the neighbouroud
    double disparity_point = maxDisp(disp,cvRound(p[0]),cvRound(p[1]),4);

    cv::Point3d p3 = getPointFromDisp(p[0],p[1],disparity_point);

    output.at<cv::Point3d>(0,i) = p3;
  }

  std::vector<cv::Point3d> output_nInf;
  std::vector<cv::Point2d> cam1pnts_nInf;

  //Before getting the error, remove the points with infinite depth 
  for (int i = 0; i < output.cols; i++)
  {
    cv::Point3d point = output.at<cv::Point3d>(0,i);
    cv::Point2d p2d = cam1pnts.at<cv::Point2d>(0,i);

    if( !std::isinf(point.x) && !std::isinf(point.y) && !std::isinf(point.z))
    {
      output_nInf.push_back(point);
      cam1pnts_nInf.push_back(p2d);
    }
  }

  output = cv::Mat(output_nInf);
  cam1pnts = cv::Mat(cam1pnts_nInf);

  cv::transpose(output, output);
  cv::transpose(cam1pnts, cam1pnts);

  return getRMS(cam1pnts,output, false);
}


static void saveXYZ(const char* filename, const cv::Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}



void DisparityExtractor::extract(const cv::Mat & image)
{

  cur_frame_ ++;
  splitVertically(image, imageleft_, imageright_);

  cv::Mat left_undist, right_undist;

  //cv::remap(imageleft_, left_undist, map11_, map12_, cv::INTER_LINEAR);
  //cv::remap(imageright_, right_undist, map21_, map22_, cv::INTER_LINEAR);

  cv::undistort(imageleft_, left_undist, cam_.intrinsics_left_, cam_.dist_left_);
  cv::undistort(imageright_, right_undist, cam_.intrinsics_right_, cam_.dist_right_);

  imageleft_ = left_undist;
  imageright_ = right_undist;

  //cv::pyrDown(imageleft_, imageleft_);
  //cv::pyrDown(imageright_, imageright_);
}

void DisparityExtractor::visualize(bool * keep_on)
{ 

  cv::Mat disp,disp8;
  cv::Mat cam0pnts,cam1pnts;

  cv::Mat pnts_left,pnts_right;

  disparity_.download(disp); 

  //disp.convertTo(disp8, CV_8U, 255/(disparter_->getNumDisparities()*16.));

  getPoints(pnts_left, pnts_right);

  //TODO: verify that the disparity is centered in the left camera
  drawPoints(pnts_left, disp);

  if(!disp.empty())
  {
    cv::namedWindow("Disparity", CV_WINDOW_AUTOSIZE);
    //cv::namedWindow("Left Camera", CV_WINDOW_AUTOSIZE);
    cv::imshow("Disparity", (cv::Mat_<uchar>)disp);
    //cv::imshow("Left Camera", imageleft_);
  }

  int k = cvWaitKey(1);
  if (k == 27)
  {
      *keep_on = false;
  }
}