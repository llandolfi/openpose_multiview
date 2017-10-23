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


struct OpenPoseParams{


	op::CvMatToOpInput *cvMatToOpInput_;
	op::CvMatToOpOutput *cvMatToOpOutput_;
	op::PoseExtractorCaffe *poseExtractorCaffeL_;
	op::PoseRenderer *poseRendererL_;
	op::OpOutputToCvMat *opOutputToCvMatL_;
	op::OpOutputToCvMat *opOutputToCvMatR_;

};


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

void emitCSV(std::ofstream & outputfile, const op::Array<float> & poseKeypoints, int camera, int cur_frame);

void filterVisible(const cv::Mat & pntsL, cv::Mat & nzL);

void filterVisible(const cv::Mat & pntsL, const cv::Mat & pntsR, cv::Mat & nzL, cv::Mat & nzR);

void filterVisible(std::vector<cv::Mat> & bodies_left, std::vector<cv::Mat> & bodies_right, double conf = 0.4);

std::string type2str(int type);

void drawPoints(const cv::Mat & points, cv::Mat & image);

void pts2VecofBodies(const cv::Mat & pts1, std::vector<cv::Mat> & bodies_left);

void vecofBodies2Pts(const std::vector<cv::Mat> bodies, cv::Mat & pts);

void splitVertically(const cv::Mat & input, cv::Mat & outputleft, cv::Mat & outputright);

double MaxPool(const cv::Mat &);

double MinPool(const cv::Mat &);

double AvgPool(const cv::Mat &);

double Pool(const cv::Mat & disp, int u, int v, int side, std::function<double(const cv::Mat &)> function);

void PoseProcess(const OpenPoseParams & params, const cv::Mat & image, op::Array<float> & poseKeypoints, cv::Mat & outputImage);
/*
* Find Correspondent bodies from camera left and right
*/
void findCorrespondences(const cv::Mat & pts1, const cv::Mat & pts2, cv::Mat & sorted_left, cv::Mat & sorted_right);

