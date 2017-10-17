#pragma once
#include "libuvc/libuvc.h"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include "utilities.hpp"
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging


struct PinholeCamera{

	PinholeCamera() {}
	PinholeCamera(const std::string params_path);

	int width_,height_;
	int fps_;

	cv::Mat intrinsics_;
	cv::Mat dist_;

};

struct DepthCamera : PinholeCamera{

	//TODO: add some parameteres 
	DepthCamera();
	DepthCamera(const std::string params_path);

	void dump();


};


struct StereoCamera {


	StereoCamera(const std::string resolution);

	void dump();

	cv::Mat getIntrinsicsLeft()
	{
		return intrinsics_left_;
	}

	std::string resolution_;
	std::string resolution_code_;
	std::string path_ = "../settings/SN1499.conf";

	int width_,height_;
	int fps_;

	cv::Mat intrinsics_left_;
	cv::Mat intrinsics_right_;
	cv::Mat RT_left_;
	cv::Mat RT_right_;
	cv::Mat dist_left_;
	cv::Mat dist_right_;

	/*Rotation matrix between the coordinate systems of the first and second cameras*/
	cv::Mat SR_;
	/*Translation vector between coordinate systems of cameras*/
	cv::Vec3d ST_;

};
