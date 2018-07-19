#include "stereo_cam.h"

std::string Camera::getResolution()
{
  return std::to_string(width_) + "x" + std::to_string(height_);
}

void PinholeCamera::JSONPoints(const cv::Mat & pnts,Json::Value & points)
{

  for(int i = 0; i < pnts.cols; i++)
  { 

      if(i % 18 >= 14)
      {
        double n = std::numeric_limits<double>::quiet_NaN();
        Json::Value jpoint;
        jpoint["x"] = n;
        jpoint["y"] = n;
        jpoint["z"] = n;
        points.append(jpoint);
      }
      else
      {
      cv::Vec3d point = pnts.at<cv::Vec3d>(0,i);
      Json::Value jpoint;
      jpoint["x"] = point[0];
      jpoint["y"] = point[1];
      jpoint["z"] = point[2];
      points.append(jpoint);
    }
  }
}

StereoCamera::StereoCamera(const std::string resolution) : resolution_(resolution)
{  }

std::ostream& operator << (std::ostream& os, const PinholeCamera & pc)
{
  
  os << pc.intrinsics_ << "\n " << pc.dist_ << "\n" << pc.height_ << "x" << pc.width_ << std::endl;

}

void StereoCamera::JSONPoints(const cv::Mat & pnts, Json::Value & points)
{
  for(int i = 0; i < pnts.cols; i++)
  { 

      if(i % 18 >= 14)
      {
        double n = std::numeric_limits<double>::quiet_NaN();
        Json::Value jpoint;
        jpoint["x"] = n;
        jpoint["y"] = n;
        jpoint["z"] = n;
        points.append(jpoint);
      }
      else
      {
      cv::Vec3d point = pnts.at<cv::Vec3d>(0,i);
      Json::Value jpoint;
      jpoint["x"] = -point[0];
      jpoint["y"] = point[1];
      jpoint["z"] = point[2];
      points.append(jpoint);
    }
  }
}

void StereoCamera::dump()
{
  std::cout << "left camera matrix"<< std::endl;
  std::cout << camera_left_.intrinsics_ << std::endl;
  std::cout << "right camera matrix"<< std::endl;
  std::cout << camera_right_.intrinsics_ << std::endl;
  std::cout << "distortion left"<< std::endl;
  std::cout << camera_left_.dist_ << std::endl;
  std::cout << "distortion right"<< std::endl;
  std::cout << camera_right_.dist_ << std::endl;
  std::cout << "Rotiation matrix" << std::endl;
  std::cout << SR_ << std::endl;
  std::cout << "Translation matrix" << std::endl;
  std::cout << ST_ << std::endl;
  std::cout << "Width " << width_ << std::endl;
  std::cout << "Height: " << height_ << std::endl;
}

PinholeCamera parsecameraJSON(const Json::Value & root)
{

  Json::Value dist = root["RadialDistortion"];
  Json::Value focallenghts = root["FocalLength"];
  Json::Value principalpoints = root["PrincipalPoint"];
  Json::Value imagesize = root["ImageSize"];

  PinholeCamera mycam;

  double fx = focallenghts[0].asDouble();
  double fy = focallenghts[1].asDouble();

  double cx = principalpoints[0].asDouble();
  double cy = principalpoints[1].asDouble();

  mycam.intrinsics_ = (cv::Mat_<double>(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

  double d1 = dist[0].asDouble();
  double d2 = dist[1].asDouble();

  mycam.dist_ = (cv::Mat_<double>(1,4) << d1, d2, 0.0, 0.0);

  mycam.height_ = imagesize[0].asInt();
  mycam.width_ = imagesize[1].asInt();

  return mycam;

}

void StereoCamera::setParameters(const std::string & paramfile)
{
  //TODO: parse camera parameters from the JSON file generated by MATLAB
  Json::Value root;
  Json::Reader reader;

  std::ifstream jfile(paramfile.c_str());
  jfile >> root;  

  Json::Value cam1 = root["CameraParameters1"];
  Json::Value cam2 = root["CameraParameters2"];

  camera_left_ = parsecameraJSON(cam1);
  camera_right_ = parsecameraJSON(cam2);

  Json::Value T = root["TranslationOfCamera2"];
  Json::Value R = root["RotationOfCamera2"];

  ST_ = cv::Vec3d(T[0].asDouble(),T[1].asDouble(),T[2].asDouble());
  ST_ = ST_ * 0.001;

  std::vector<cv::Vec3d> rows;
  for(int i = 0; i < 3 ; i++)
  {
     Json::Value row = R[i];
     rows.push_back(cv::Vec3d(row[0].asDouble(),row[1].asDouble(),row[2].asDouble()));
  }

  SR_ = (cv::Mat_<double>(3,3) << rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], 
          rows[2][0], rows[2][1], rows[2][2]);
}

DepthCamera::DepthCamera()
{
  width_ = 480;
  height_ = 640;

  intrinsics_ = (cv::Mat_<double>(3,3) << 585.187492217609, 0, 322.714077555293, 0, 585.308616340665, 248.626108676666, 0, 0, 1);
}

int DepthCamera::getWidth()
{
  return height_;
}

int DepthCamera::getHeight()
{
  return width_;
}

DepthCamera::DepthCamera(const std::string params_path)
{

  cv::FileStorage fs(params_path, cv::FileStorage::READ);
  fs["rgb_intrinsics"] >> intrinsics_;

  height_ = fs["image_height"];
  width_ = fs["image_width"];

}

std::string DepthCamera::getResolution()
{
  return std::to_string(height_) + "x" + std::to_string(width_);
}

void DepthCamera::dump()
{
  std::cout << "intrinsics matrix " << std::endl;
  std::cout << intrinsics_ << std::endl;
}

Kinect2::Kinect2(bool full_hd)
{ 
  if (full_hd)
  {
    width_ = 1920;
    height_ = 1080;

    intrinsics_ = (cv::Mat_<double>(3,3) << 1039.7114637900604, 0.0, 937.6437462083773, 0.0, 1040.464051222892, 526.146628446275,0.0,0.0,1.0);
  }
  else 
  {
    width_ = 512;
    height_ = 424;

    intrinsics_ = (cv::Mat_<double>(3,3) << 368.096588, 0.0, 261.696594, 0.0, 368.096588, 202.522202,0.0,0.0,1.0);
  }
}


std::string Kinect2::getResolution()
{
  return std::to_string(width_) + "x" + std::to_string(height_);
}

int Kinect2::getHeight()
{
  return height_;
}

int Kinect2::getWidth()
{
  return width_;
}


ZED::ZED(const std::string resolution) : resolution_(resolution)
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
    if (lines[i].find("LEFT_"+resolution_code_) < lines[i].size())
    { 

      std::cout << lines[i] << std::endl;
      std::cout << "Found at " << lines[i] << std::endl;
      std::cout << lines[i].find(resolution_code_) << std::endl;

      fx = getDouble(lines[i+3],"=");
      fy = getDouble(lines[i+4],"=");
      cx = getDouble(lines[i+1],"=");
      cy = getDouble(lines[i+2],"=");
      
      intrinsics.at<double>(0,0) = fx;
      intrinsics.at<double>(0,2) = cx;
      intrinsics.at<double>(1,1) = fy;
      intrinsics.at<double>(1,2) = cy;
      std::cout << "intrinsics left " << std::endl;
      std::cout << intrinsics << std::endl;

      camera_left_.intrinsics_ = intrinsics.clone();

      dist_coeffs.at<double>(0,0) = getDouble(lines[i+5],"=");
      dist_coeffs.at<double>(0,1) = getDouble(lines[i+6],"=");

      std::cout << "left distortion " << std::endl;
      std::cout << dist_coeffs << std::endl;
      camera_left_.dist_ = dist_coeffs.clone();

    }

    if (lines[i].find("RIGHT_"+resolution_code_) < lines[i].size())
    { 

      std::cout << lines[i] << std::endl;
      std::cout << "Found at " << lines[i] << std::endl;
      std::cout << lines[i].find(resolution_code_) << std::endl;

      fx = getDouble(lines[i+3],"=");
      fy = getDouble(lines[i+4],"=");
      cx = getDouble(lines[i+1],"=");
      cy = getDouble(lines[i+2],"=");
      
      intrinsics.at<double>(0,0) = fx;
      intrinsics.at<double>(0,2) = cx;
      intrinsics.at<double>(1,1) = fy;
      intrinsics.at<double>(1,2) = cy;

      std::cout << "intrinsics right " << std::endl;
      std::cout << intrinsics << std::endl;

      camera_right_.intrinsics_ = intrinsics.clone();

      dist_coeffs.at<double>(0,0) = getDouble(lines[i+5],"=");
      dist_coeffs.at<double>(0,1) = getDouble(lines[i+6],"=");

      camera_right_.dist_ = dist_coeffs.clone();

      std::cout << "right distortion " << std::endl;
      std::cout << dist_coeffs << std::endl;

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

  fps_ = getZEDfps();

}

int ZED::getZEDfps()
{
  if(strcmp(resolution_.c_str(),"2208x1242") == 0)
  {
    return 15;
  }

  if(strcmp(resolution_.c_str(),"1920x1080") == 0)
  {
    return 30;
  }

  if(strcmp(resolution_.c_str(), "1280x720") == 0)
  {
    return 60;
  }

  if(strcmp(resolution_.c_str(), "672x376") == 0)
  {
    return 100;
  }

  return -1;
}

void ZED::JSONPoints(const cv::Mat & pnts, Json::Value & points)
{

  for(int i = 0; i < pnts.cols; i++)
  { 

      if(i % 18 >= 14)
      {
        double n = std::numeric_limits<double>::quiet_NaN();
        Json::Value jpoint;
        jpoint["x"] = n;
        jpoint["y"] = n;
        jpoint["z"] = n;
        points.append(jpoint);
      }
      else
      {
      cv::Vec3d point = pnts.at<cv::Vec3d>(0,i);
      Json::Value jpoint;
      jpoint["x"] = point[0];
      jpoint["y"] = -point[1];
      jpoint["z"] = point[2];
      points.append(jpoint);
    }
  }
}