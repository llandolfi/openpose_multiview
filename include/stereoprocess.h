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
#include "stereo_cam.h"
#include <opencv2/cudastereo.hpp>
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging

extern bool keep_on;


void splitVertically(const cv::Mat & input, cv::Mat & outputleft, cv::Mat & outputright);
std::vector<std::string> CSVTokenize(std::string kpl_str);
void emitCSV(std::ofstream & outputfile, std::string & kp_str, const op::Array<float> & poseKeypoints, int camera);

struct PoseExtractor {

	PoseExtractor(int argc, char **argv, const std::string resolution);

	virtual double go(const cv::Mat & image, const bool verify, cv::Mat &, bool* keep_on);

	virtual double triangulate(cv::Mat &)=0;

	virtual void process()=0;

	virtual void extract(const cv::Mat &)=0;

	virtual void visualize(bool* keep_on)=0;

	virtual void verify(const cv::Mat & pnts, bool* keep_on)=0;

	virtual void setDepth(const cv::Mat & m);

	virtual void init();

	virtual void destroy();

	op::Array<float> poseKeypointsL_;

	cv::Mat outputImageL_;

	op::CvMatToOpInput *cvMatToOpInput_;
	op::CvMatToOpOutput *cvMatToOpOutput_;
	op::PoseExtractorCaffe *poseExtractorCaffeL_;
	op::PoseRenderer *poseRendererL_;
	op::OpOutputToCvMat *opOutputToCvMatL_;
	op::OpOutputToCvMat *opOutputToCvMatR_;

	bool inited_;

	cv::VideoWriter outputVideo_; 
	std::ofstream outputfile_;   

	int cur_frame_;	

	PinholeCamera * pcam_;

	cv::Mat depth_;
};

struct DepthExtractor : PoseExtractor {

	DepthExtractor(int argc, char **argv, const std::string resolution);

	virtual double triangulate(cv::Mat &);

	virtual void process();

	virtual void extract(const cv::Mat &);

	virtual void visualize(bool* keep_on);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);

	cv::Point3d getPointFromDepth(double u, double v, double z);

	cv::Mat RGB_;

};


struct StereoPoseExtractor : PoseExtractor {

	StereoPoseExtractor(int argc, char **argv, const std::string resolution);

	void triangulateCore(cv::Mat & cam0pnts, cv::Mat & cam1pnts, cv::Mat & finalpoints);

	void parseIntrinsicMatrix(const std::string path = "../settings/SN1499.conf");

	virtual void getPoints(cv::Mat &, cv::Mat &);

	virtual double triangulate(cv::Mat &);

	virtual void visualize(bool* keep_on);

	virtual void process();

	virtual void extract(const cv::Mat &);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);

	virtual double getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D, bool left = true);

	op::Array<float> poseKeypointsR_;

	cv::Mat imageleft_;
	cv::Mat imageright_;
	cv::Mat outputImageR_;

	StereoCamera cam_;

};

struct DisparityExtractor : StereoPoseExtractor {

	DisparityExtractor(int argc, char **argv, const std::string resolution);

	void getDisparity();

	cv::Point3d getPointFromDisp(double u, double v, double d);

	double maxDisp(const cv::Mat & disp, int u, int v, int side);

	double avgDisp(const cv::Mat & disp, int u, int v, int side = 5);

	void verifyD(const cv::Mat & pnts, bool* keep_on);

	virtual void extract(const cv::Mat & image);

	virtual double triangulate(cv::Mat & output); 

	virtual void visualize(bool * keep_on);


	cv::cuda::GpuMat disparity_;
	cv::cuda::GpuMat gpuleft_,gpuright_;

	cv::Mat P_;
	cv::Mat iP_;
	cv::Mat Q_;
	cv::Rect roi1_, roi2_;
	cv::Mat map11_, map12_, map21_, map22_;

	cv::Ptr<cv::cuda::StereoBM> disparter_ = cv::cuda::createStereoBM(128,3);

	int ndisp,iters,levels = 0;

	//BELIEF PROPAGATION WORKS BETTER BUT MUCH SLOWER
	//cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(1280,720,&ndisp,&iters,&levels);
	//cv::Ptr<cv::cuda::StereoBeliefPropagation> disparter_ = cv::cuda::createStereoBeliefPropagation(128,3);

};

struct PoseExtractorFromFile : StereoPoseExtractor {

	PoseExtractorFromFile(int argc, char **argv, const std::string resolution, const std::string path);
                                        
	virtual void process(const cv::Mat & image);

	virtual void visualize(bool * keep_on);

	virtual void getPoints(cv::Mat & outputL, cv::Mat & outputR);

	void getNextBlock(std::vector<std::vector<std::string>> & lines);

	const std::string filepath_;
	std::ifstream file_;
	std::string line_;

	std::vector<cv::Point2d> points_left_;
	std::vector<cv::Point2d> points_right_;

};