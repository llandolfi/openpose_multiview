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
#include <json/json.h>

struct Camera
{
	Camera(){}

	Camera(const cv::Mat intrinsics, const cv::Mat & distortion, int width, int height) : width_(width), height_(height), 
																						intrinsics_(intrinsics), dist_(distortion)
	{}

	int width_,height_;
	int fps_;

	cv::Mat intrinsics_;
	cv::Mat dist_;

	virtual std::string getResolution();
	virtual void JSONPoints(const cv::Mat & pnts,Json::Value & points) = 0;
};


struct PinholeCamera : Camera { 

	PinholeCamera() {}
	PinholeCamera(const std::string params_path);

	virtual void JSONPoints(const cv::Mat & pnts,Json::Value & points);

	friend std::ostream& operator << (std::ostream& os, const PinholeCamera & pc);

};

struct DepthCamera : PinholeCamera{

	//TODO: add some parameteres 
	DepthCamera();
	DepthCamera(const std::string params_path);

	//void JSONPoints(const cv::Mat & pnts,Json::Value & points);

	void dump();

	std::string getResolution();


};


struct StereoCamera : Camera {

	StereoCamera(){}
	StereoCamera(const std::string resolution);

	void dump();
	void setParameters(const std::string & paramfile);
	virtual void JSONPoints(const cv::Mat & pnts,Json::Value & points);

	std::string resolution_;

	PinholeCamera camera_left_;
	PinholeCamera camera_right_;

	/*Rotation matrix between the coordinate systems of the first and second cameras*/
	cv::Mat SR_;
	/*Translation vector between coordinate systems of cameras*/
	cv::Vec3d ST_;
};

struct ZED : StereoCamera {

	ZED(const std::string resolution);

	int getZEDfps();
	void JSONPoints(const cv::Mat & pnts,Json::Value & points);

	std::string resolution_code_;
	std::string path_ = "../settings/SN1499.conf";	
	std::string resolution_;

};

PinholeCamera parsecameraJSON(const Json::Value & root);


