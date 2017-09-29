#include "stereoprocess.h"
#include "utilities.hpp"

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
  double b = -cam_.ST_[0];

  double Z = (f * b)/d;
  double X = (u * Z)/f;
  double Y = (v * Z)/f;

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

  //TODO: be sure that depth measures are exact if disparity is taken 

  for(int i=0; i < cam0pnts.cols; i++)
  {

    cv::Vec2d p = cam0pnts.at<cv::Vec2d>(0,i);
    double dispatpoint = (double)disp.at<uint16_t>(cvRound(p[0]),cvRound(p[1]));
    double disparity_point = maxDisp(disp,cvRound(p[0]),cvRound(p[1]),5);
    std::cout << "Disparity " << disparity_point << std::endl;
    cv::Point3d p3 = getPointFromDisp(p[0],p[1],disparity_point);
    std::cout << "Point " << p3 << std::endl;
    output.at<cv::Point3d>(0,i) = p3;
  }


  //TODO: get the error (project the point3D in camera right)

  std::cout << "Error: " << getRMS(cam0pnts, output, false) << std::endl;
  return getRMS(cam0pnts,output, false);
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


void DisparityExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 

  if(pnts.empty())
  {
    return;
  }
  
  std::cout << "points to be projected " << std::endl;
  std::cout << pnts << std::endl;
  std::cout << "intrinsics " << std::endl;
  std::cout << cam_.intrinsics_left_ << std::endl;

  std::vector<cv::Point2d> points2D(pnts.cols);

  cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_.intrinsics_left_,cam_.dist_right_,points2D);

  //TODO: write circles in projected points
  cv::Mat verification = imageleft_.clone();
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