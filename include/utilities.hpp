#pragma once
#include "libuvc/libuvc.h"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
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
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging


int getHeight(const std::string & resolution);

int getWidth(const std::string & resolution);

int getInt(const std::string & s, const std::string c);

double getDouble(const std::string & s, const std::string c);

constexpr unsigned int str2int(const char* str, int h = 0);

const std::string getResolutionCode(const std::string resolution);

cv::Point2d project(const cv::Mat & intrinsics, const cv::Vec3d & p3d);

void vector2Mat(const std::vector<cv::Point2d> & points, cv::Mat & pmat);

void opArray2Mat(const op::Array<float> & keypoints, cv::Mat & campnts);

std::vector<std::string> CSVTokenize(std::string kpl_str);

void emitCSV(std::ofstream & outputfile, std::string & kp_str, const op::Array<float> & poseKeypoints, int camera, int cur_frame);

void filterVisible(const cv::Mat & pntsL, cv::Mat & nzL);

void filterVisible(const cv::Mat & pntsL, const cv::Mat & pntsR, cv::Mat & nzL, cv::Mat & nzR);

std::string type2str(int type);

void drawPoints(const cv::Mat & points, cv::Mat & image);

