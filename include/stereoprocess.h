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
#include "netutils.hpp"
#include <limits>
#include <chrono>
#include <opencv2/cudastereo.hpp>
#include "image_frame.hpp"
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging


extern bool keep_on;


struct PoseExtractor {

	PoseExtractor(int argc, char **argv, Camera & camera);

	virtual double go(const ImageFrame & image, const bool verify, cv::Mat &, bool* keep_on);

	virtual double triangulate(cv::Mat &)=0;

	virtual bool track()=0;

	virtual void process(const std::string & write_keypoint, bool visualize);

	virtual void extract(const ImageFrame &)=0;

	virtual void appendFrame(const ImageFrame &)=0;

	virtual void visualize(bool* keep_on)=0;

	virtual void verify(const cv::Mat & pnts, bool* keep_on)=0;

	virtual void prepareVideo(const std::string & path)=0;

	virtual void prepareOutputVideo(const std::string & path)=0;

	virtual double computeTrackError()=0;

	virtual std::string pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time)=0;

	virtual void outputPoints(const cv::Mat & points3D);

	virtual void finalize(){}

	virtual void setDepth(const cv::Mat & m);

	virtual void init();

	virtual void destroy();

	cv::Mat imageleft_;

	OpenPoseParams pose_params_;

	cv::Mat outputImageL_;
	op::Array<float> poseKeypointsL_;

	bool inited_;

	cv::VideoWriter outputVideo_; 
	cv::VideoWriter poseVideo_;
	std::ofstream outputfile_; 
	std::ofstream outputfile3D_; 

	std::ofstream timefile_;  
	std::ofstream jsonfile_;

	std::ofstream error_file_;

	int cur_frame_;	
	int skip_ = 0;

	cv::Mat depth_;

	bool live_ = true;
	bool videooutput_ = false;

	bool tracking2D_ = false;
	bool tracking3D_ = false;
	bool tracked_ = false;
	cv::Mat trackedpnts_;

	std::string videoname_ = "";

	UDPStreamer udpstreamer_;

	cv::Mat prev_gray_, gray_;
	std::vector<cv::Point2f> points_[2];
	cv::Size winSize_;

};

struct DepthExtractor : PoseExtractor {

	DepthExtractor(int argc, char **argv, DepthCamera & camera, const std::string & depth_video);

	virtual double triangulate(cv::Mat &);

	virtual bool track();

	virtual void process(const std::string & write_keypoint, bool visualize);

	virtual void extract(const ImageFrame &);

	virtual void visualize(bool* keep_on);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);

	virtual void prepareVideo(const std::string & path);

	virtual void prepareOutputVideo(const std::string & path);

	virtual void encodeDepth(const cv::Mat & depth, cv::Mat & output);

	virtual void decodeDepth(const cv::Mat & rgb, cv::Mat & depth);

	virtual double getDepthPoint(int x, int y);

	void finalize();

	double getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D);

	void appendFrame(const ImageFrame &);

	cv::Point3d getPointFromDepth(double u, double v, double z);

	std::string pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time);

	void kernel2CSV(int idx, const cv::Mat & kernel);

	double computeTrackError();

	cv::Mat RGB_;

	cv::VideoWriter depthoutput_; 

	cv::VideoCapture depthcap_;

	std::string depthpath_;

	std::string kernel_output_;
	std::ofstream kernelcsv_;
	

	DepthCamera cam_;

	bool fframe_ = true;
};

struct ONIDepthExtractor : DepthExtractor {

	ONIDepthExtractor(int argc, char**argv, DepthCamera & camera, const std::string & depth_video);

	void encodeDepth(const cv::Mat & depth, cv::Mat & output);

	void decodeDepth(const cv::Mat & rgb, cv::Mat & depth);

	void prepareVideo(const std::string & path);

	void appendFrame(const ImageFrame & myframe);

	void extract(const ImageFrame & m);

	std::ofstream out_oni_;
	uint8_t* in_oni_;

	uint64_t pos = 0;

};

struct Depth2Extractor : DepthExtractor {

	Depth2Extractor(int argc, char**argv, DepthCamera & camera, const std::string & depth_video);

	double getDepthPoint(int x, int y);
};

struct ONIDepth2Extractor : ONIDepthExtractor {

	ONIDepth2Extractor(int argc, char**argv, DepthCamera & camera, const std::string & depth_video);

	double getDepthPoint(int x, int y);
};


struct StereoPoseExtractor : PoseExtractor {

	StereoPoseExtractor(int argc, char **argv, StereoCamera & camera);

	void triangulateCore(cv::Mat & cam0pnts, cv::Mat & cam1pnts, cv::Mat & finalpoints);

	void parseIntrinsicMatrix(const std::string path = "../settings/SN1499.conf");

	std::string pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time);

	double computeTrackError();

	virtual void getPoints(cv::Mat &, cv::Mat &);

	virtual double triangulate(cv::Mat &);

	virtual bool track();

	virtual void visualize(bool* keep_on);

	virtual void process(const std::string & write_keypoint, bool visualize);

	virtual void extract(const ImageFrame &);

	virtual void appendFrame(const ImageFrame &);

	virtual void verify(const cv::Mat & pnts, bool* keep_on);

	virtual void prepareVideo(const std::string & path);

	virtual void prepareOutputVideo(const std::string & path);

	virtual double getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D, bool left = true);

	void outputPoints(const cv::Mat & points3D);

	op::Array<float> poseKeypointsR_;
	cv::Mat trackedpntsR_;

	cv::Mat outputImageR_;

	cv::Mat imageright_;

	StereoCamera * cam_;

	cv::Mat prev_grayR_, grayR_;
	std::vector<cv::Point2f> pointsR_[2];

	bool tracked_left_ = false;
	bool tracked_right_ = false;

};

struct DisparityExtractor : StereoPoseExtractor {

	DisparityExtractor(int argc, char **argv, StereoCamera & camera);

	void getDisparity();

	cv::Point3d getPointFromDisp(double u, double v, double d);

	double maxDisp(const cv::Mat & disp, int u, int v, int side);

	double avgDisp(const cv::Mat & disp, int u, int v, int side = 5);

	void verifyD(const cv::Mat & pnts, bool* keep_on);

	std::string pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time);

	virtual void extract(const ImageFrame & image);

	virtual double triangulate(cv::Mat & output); 

	virtual bool track();

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

	PoseExtractorFromFile(int argc, char **argv, StereoCamera & camera, const std::string path);
                                      
	virtual void visualize(bool * keep_on);

	virtual void getPoints(cv::Mat & outputL, cv::Mat & outputR);

	void process(const cv::Mat & image);

	void getNextBlock(std::vector<std::vector<std::string>> & lines);

	const std::string filepath_;
	std::ifstream file_;
	std::string line_;

	std::vector<cv::Point2d> points_left_;
	std::vector<cv::Point2d> points_right_;

};