#include "stereo_cam.h"

StereoCamera::StereoCamera(const std::string resolution) : resolution_(resolution)
{ 

  std::cout << "resolution " << resolution_ << std::endl;
  resolution_code_ = getResolutionCode(resolution_);
  std::cout << "resolution code " << resolution_code_ << std::endl;
  
  width_ = getWidth(resolution_);
  height_ = getHeight(resolution_);


  std::ifstream infile(path_);
  std::string line;
  double fx,fy,cx,cy = 0;
  double k1,k2 = 0.0;
  cv::Mat intrinsics = cv::Mat::eye(3,3,CV_64F);
  cv::Mat dist_coeffs = cv::Mat::zeros(8,1,CV_64F);

  std::vector<std::string> lines;
  std::copy(std::istream_iterator<std::string>(infile),std::istream_iterator<std::string>(),back_inserter(lines));

  for (int i = 0; i < lines.size(); i ++)
  { 
    if (lines[i].find(resolution_code_) < lines[i].size())
    { 

      std::cout << "Found at " << lines[i] << std::endl;
      std::cout << lines[i].find(resolution_code_) << std::endl;

      fx = getDouble(lines[i+1],"=");
      fy = getDouble(lines[i+2],"=");
      cx = getDouble(lines[i+3],"=");
      cy = getDouble(lines[i+4],"=");
      
      intrinsics.at<double>(0,0) = fx;
      intrinsics.at<double>(0,2) = cx;
      intrinsics.at<double>(1,1) = fy;
      intrinsics.at<double>(1,2) = cy;

      intrinsics_left_ = intrinsics.clone();

      dist_coeffs.at<double>(0,0) = getDouble(lines[i+5],"=");
      dist_coeffs.at<double>(0,1) = getDouble(lines[i+6],"=");

      dist_left_ = dist_coeffs.clone();

      i = i + 8;

      fx = getDouble(lines[i+1],"=");
      fy = getDouble(lines[i+2],"=");
      cx = getDouble(lines[i+3],"=");
      cy = getDouble(lines[i+4],"=");
      
      intrinsics.at<double>(0,0) = fx;
      intrinsics.at<double>(0,2) = cx;
      intrinsics.at<double>(1,1) = fy;
      intrinsics.at<double>(1,2) = cy;

      intrinsics_right_ = intrinsics.clone();

      dist_coeffs.at<double>(0,0) = getDouble(lines[i+5],"=");
      dist_coeffs.at<double>(0,1) = getDouble(lines[i+6],"=");

      dist_right_ = dist_coeffs.clone();

      break;

    }
  }

  double baseline = 0;
  cv::Vec3d rotv;
  std::string quality = resolution_code_.substr(resolution_code_.find("_") + 1);

  int count = 0;

  for(int i = 0; i < lines.size(); i++)
  {
    if(lines[i].find("STEREO") < lines[i].size())
    {

      baseline = getDouble(lines[i+1],"=");

      int j = i +1;

      while(count < 3)
      {
        if (lines[j].find("_" + quality) < lines[j].size())
        {
          rotv[count] = getDouble(lines[j],"=");
          count ++;
        }

        j++;
      }
      break;
    }
  }

  ST_ = cv::Vec3d(-baseline*0.001, 0.0, 0.0);

  double tmp = rotv[0];
  rotv[0] = rotv[1];
  rotv[1] = tmp;

  cv::Rodrigues(rotv,SR_);

  std::cout << "stereo rotation " << SR_ << std::endl;
  std::cout << "stereo translation " << ST_ << std::endl;
}

void StereoCamera::dump()
{
  std::cout << "left camera matrix"<< std::endl;
  std::cout << intrinsics_left_ << std::endl;
  std::cout << "right camera matrix"<< std::endl;
  std::cout << intrinsics_right_ << std::endl;
  std::cout << "distortion left"<< std::endl;
  std::cout << dist_left_ << std::endl;
  std::cout << "distortion right"<< std::endl;
  std::cout << dist_right_ << std::endl;
  std::cout << "Rotiation matrix" << std::endl;
  std::cout << SR_ << std::endl;
  std::cout << "Translation matrix" << std::endl;
  std::cout << ST_ << std::endl;
  std::cout << "Width " << width_ << std::endl;
  std::cout << "Height: " << height_ << std::endl;
}