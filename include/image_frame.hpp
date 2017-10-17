#pragma once
#include <chrono>
#include <opencv2/core/core.hpp>
#include <string>
#include <json/json.h>

struct CamParams 
{
  CamParams(cv::Mat cam_matrix, cv::Mat dist, unsigned int width,
            unsigned int height)
      : cam_matrix_(cam_matrix), dist_(dist), width_(width), height_(height) {}

  CamParams() {}

  cv::Mat cam_matrix_, dist_;
  unsigned int width_ = 0;
  unsigned int height_ = 0;

  void toJSON(Json::Value & root);
};

struct ImageFrame {
   enum class DepthRegistration { None, Undistorted, ColorToDepth, DepthToColor };

  bool hasColor() const { return !color_.empty(); }
  bool hasDepth() const { return !depth_.empty(); }

  static const char * mode2str(DepthRegistration x)
  {
  	const char * cp[4] = {"none","undistorted","colortodepth","depthtocolor"};
  	return cp[(int)x];
  }

  DepthRegistration depthmode_ = DepthRegistration::None; 
  cv::Mat color_, depth_;
  std::string type_;
  CamParams color_params_;
  CamParams depth_params_;
  long seq_number_;

  std::chrono::high_resolution_clock::time_point time_stamp_;
};



inline void CamParams::toJSON(Json::Value & root)
{
	Json::Value intrinsics = Json::arrayValue;

	for(size_t i = 0; i < 3; ++i)
		for(size_t j = 0; j < 3; ++j)
			intrinsics.append(cam_matrix_.at<float>(i,j));
	root["intrinsics"] = intrinsics;

	Json::Value dist = Json::arrayValue;

	for(int i = 0; i < dist_.rows; ++i)
			dist.append(dist_.at<float>(i,0));
	root["dist"] = dist;
	root["width"] = width_;
	root["height"] = height_;

}
