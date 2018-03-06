#include "stereoprocess.h"
#include "opencv2/cudastereo.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>
#include <bitset>
#include <sys/mman.h>
#include "xn16zdec.h"
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>

DEFINE_string(kernel_output,              "",      "Path of the kernel output file");
//DEFINE_bool(ramcip,              false,            "set to true if depth data from ramcip dataset");


std::map<int, std::string> body_map = {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Background"},
};


bool inited = false;


DepthExtractor::DepthExtractor(int argc, char **argv, DepthCamera & camera, const std::string & depth_video) : PoseExtractor(argc, argv, camera)
{

  cam_ = camera;
  depthpath_ = depth_video;
  kernel_output_ = FLAGS_kernel_output;

  if(kernel_output_ != "")
  {
    kernelcsv_.open(kernel_output_);
  }
}

Depth2Extractor::Depth2Extractor(int argc, char **argv, DepthCamera & camera, const std::string & depth_video) : 
  DepthExtractor(argc, argv, camera, depth_video)
{}

ONIDepth2Extractor::ONIDepth2Extractor(int argc, char **argv, DepthCamera & camera, const std::string & depth_video) : 
  ONIDepthExtractor(argc, argv, camera, depth_video)
{}


void DepthExtractor::finalize()
{

}

std::string DepthExtractor::pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time)
{

  Json::Value points;
  Json::Value colors;


  for(int i = 0; i < pnts.cols; i++)
  { 
    cv::Vec3d point = pnts.at<cv::Vec3d>(0,i);
    Json::Value jpoint;
    jpoint["x"] = point[0];
    jpoint["y"] = -point[1];
    jpoint["z"] = point[2];
    points.append(jpoint);
  }

  //cam_->JSONPoints(pnts,points);
  
  colors["r"] = 1.0;
  colors["g"] = 0;
  colors["b"] = 0;
  

  Json::Value root;
  root["type"] = "bodypoints";
  root["frame"] = frame;
  root["id"] = "uniquestring";
  root["radius"] = 0.08;
  root["pointorder"] = "openpose";
  root["color"] = colors;
  root["points"] = points;
  root["timestamp"] = time;

  Json::FastWriter writer;
  Json::StyledWriter writerp;

  //std::cout << "sending: " << std::endl;
  //std::cout << writerp.write(root) << std::endl;

  return writer.write(root);
}

/*
* x: point coordinate in pixel
* y: point coordinate in pixel
* d: disparity at point (x,y)
*/
cv::Point3d DepthExtractor::getPointFromDepth(double u, double v, double z)
{

  if(z == 0.0)
  {
    return cv::Point3d(0,0,0);
  }

  double fx = cam_.intrinsics_.at<double>(0,0);
  double fy = cam_.intrinsics_.at<double>(1,1);
  double cx = cam_.intrinsics_.at<double>(0,2);
  double cy = cam_.intrinsics_.at<double>(1,2);

  double Z = z;
  double X = ((v - cx) * Z)/fx;
  double Y = ((u - cy) * Z)/fy;

  return cv::Point3d(X,Y,Z);

}

double DepthExtractor::getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D)
{
  if(pnts3D.empty())
  {
    return 0.0;
  }

  cv::Mat points2D; 

  cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_.intrinsics_,cv::Vec4d(0,0,0,0),points2D);

  cv::transpose(points2D,points2D);

  return cv::norm(points2D - cam0pnts);
}

void DepthExtractor::kernel2CSV(int idx, const cv::Mat & kernel)
{
  //TODO: get current frame, get body part from i, write kernel left to right up to bottom
  std::vector<int> depths;
  int frame = cur_frame_;
  std::string bodypart = body_map[idx];

  for(int i=0; i < kernel.rows; i++)
  {
    for(int j=0; j < kernel.cols; j++)
    {
      depths.push_back((int)kernel.at<uint16_t>(i,j));
    }
  }

  kernelcsv_ << cur_frame_ << " " << idx << " ";

  for(auto &a : depths)
  {
    kernelcsv_ << a << " ";
  }

  kernelcsv_ << "\n";
}

int mostConfident(const cv::Mat & bp)
{

  std::vector<double> confidences(bp.cols/18);

  for (int i = 0; i < bp.cols/18; i++)
  { 
    double sumconfidence = 0.0;

    for(int j = 0; j < 18; j++)
    {
      sumconfidence += bp.at<cv::Vec3d>((18*i)+j)[2];
    }

    confidences[i] = sumconfidence;
  }

  return *std::max_element(confidences.begin(), confidences.end());
}

bool DepthExtractor::track()
{ 

  bool trackerror = true;

  if(cur_frame_ % 10 == 1)
  {
    points_[0].clear();
    return false;
  }

  if(points_[0].size() == 0)
  {
   cv::Mat bodypartsL;
   opArray2Mat(poseKeypointsL_, bodypartsL);
   mat2Vector(bodypartsL,points_[0]);
  }

  bool nclear = trackLK(prev_gray_, gray_, points_[0], points_[1], 2.5, trackedpnts_);

  if(!nclear)
  {
    points_[0].clear();
  }

  if(nclear)
  {
    prev_gray_ = gray_.clone();
  }
  
  return nclear;
}

double DepthExtractor::computeTrackError()
{
  cv::Mat cam0pnts;
  opArray2Mat(poseKeypointsL_, cam0pnts);
  
  return computeTrackErrorU(cam0pnts, trackedpnts_);
}

double DepthExtractor::getDepthPoint(int x, int y)
{
  return (double)depth_.at<uint16_t>(x, y); 
}

double DepthExtractor::triangulate(cv::Mat & finalpoints)
{ 

  double epsilon = 100;
  cv::Mat cam0pnts;
  double confidence = 0.0;


  if(!tracked_)
  {
    opArray2Mat(poseKeypointsL_, cam0pnts);
  }
  else
  { 
    cam0pnts = trackedpnts_;
  }


  if(cam0pnts.empty())
  {
    return 0.0;
  }

  std::vector<cv::Point3d> points3D;
  std::vector<cv::Point2d> points2D;

   
  //filterVisible(cam0pnts, cam0pnts);

  //TODO: get only the most confident body if needed
  //int mc = mostConfident(cam0pnts);  

  for( int i = 0; i < cam0pnts.cols; i++)
  {

    cv::Point3d pwithnot = cam0pnts.at<cv::Point3d>(0,i);
    confidence = pwithnot.z;
    cv::Point2d keypoint(cvRound(pwithnot.y), cvRound(pwithnot.x));

    ///std::cout << "Keypoints " << poseKeypointsL_.toString() << std::endl;
    //std::cout << "rows: " << depth_.rows << " columns: " << depth_.cols << std::endl;

    if(keypoint.x > RGB_.rows)
    {
      std::cout << "Attenzione x " << keypoint.x << std::endl;
      keypoint.x = (double)RGB_.rows-1.0;
      //exit(-1);
    }

    if(keypoint.y > RGB_.cols)
    {
      std::cout << "Attenzione y " << keypoint.y << std::endl;
      keypoint.y = (double)RGB_.cols-1.0;
      //exit(-1);
    }

    cv::Mat kernel;
    cv::Point3d point = getPointFromDepth(keypoint.x,keypoint.y,
                        getDepthPoint(keypoint.x, keypoint.y));
                        //(double)depth_.at<uint16_t>(keypoint.x, keypoint.y));
                        //Pool(depth_, keypoint.y, keypoint.x, 7, gaussianAvg, kernel));

    //uint16_t ddepth = depth_.at<uint16_t>(keypoint.y, keypoint.x);

   //std::cout << "depth: " << point.z  << std::endl;

    if(FLAGS_kernel_output != "")// && i/18 == mc)
    {
      kernel2CSV(i,kernel);
    }

    if(tracked_)
    {
      if(point.x != 0.0 && point.y != 0.0 && point.z != 0.0)
      {
        point = point / 1000;
        points3D.push_back(point);
        points2D.push_back(keypoint);
      }
      else
      { 
        double n = std::numeric_limits<double>::quiet_NaN();
        points3D.push_back(cv::Point3d(n,n,n));
        points2D.push_back(keypoint);
      }
    }
    else
    {
      if(point.x != 0.0 && point.y != 0.0 && point.z != 0.0 && confidence > 0.2)
      {
        point = point / 1000;
        points3D.push_back(point);
        points2D.push_back(keypoint);
      }
      else
      { 
        double n = std::numeric_limits<double>::quiet_NaN();
        points3D.push_back(cv::Point3d(n,n,n));
        points2D.push_back(keypoint);
      }
    }
  }

  //TODO: remove zeros also from points3D and the correspondent from points 2D
  cam0pnts = cv::Mat(points2D);
  cv::transpose(cam0pnts, cam0pnts);

  cv::Mat tmp = cv::Mat(points3D);
  finalpoints = tmp.clone();
  cv::transpose(finalpoints, finalpoints);

  double error = getRMS(cam0pnts, finalpoints);

  return error;
}

/**
* Converts depth into RGB image for MPEG encoding
*/
void DepthExtractor::encodeDepth(const cv::Mat & depth, cv::Mat & output)
{
  cv::Mat dc1 = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  cv::Mat dc2 = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  cv::Mat dummy = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);

  for (int i=0; i < depth.rows; i++)
  {
    for(int j=0; j < depth.cols; j++)
    {
      uint16_t val = depth.at<uint16_t>(i,j);
      uint8_t * ptr = (uint8_t*)&val;

      dc1.at<uint8_t>(i,j) = ptr[1];
      dc2.at<uint8_t>(i,j) = ptr[0];
    }
  }

  std::vector<cv::Mat> channels;
  channels.push_back(dc1);
  channels.push_back(dc2);
  channels.push_back(dummy);

  cv::merge(channels, output);
}

/**
* Decodes an encoded depth frame from kinect
*/
void DepthExtractor::decodeDepth(const cv::Mat & rgb, cv::Mat & depth)
{ 

  depth = cv::Mat(rgb.rows, rgb.cols, CV_16UC1);

  uint8_t buf[2];
  uint16_t * decodecptr = (uint16_t*)&buf;

  for (int i=0; i < depth.rows; i++)
  {
    for(int j=0; j < depth.cols; j++)
    {
      cv::Vec3b value = rgb.at<cv::Vec3b>(i,j);
      //std::cout << "value "<< value << std::endl;

      buf[0] = value[1];
      buf[1] = value[0];
      
      depth.at<uint16_t>(i,j) = *decodecptr;
    }
  }
}

void DepthExtractor::appendFrame(const ImageFrame & myframe)
{

 outputVideo_ << myframe.color_;

 //try to show myframe.color

 
 //convert depth in normal codec: from single channel 16 bit 
 //cv::Mat depthtosave;

 cv::Mat depthtosave;
 encodeDepth(myframe.depth_, depthtosave);

 depthoutput_ << depthtosave;  

 timefile_ << std::to_string(myframe.time_stamp_.count()) << "\n";
}


void ONIDepthExtractor::appendFrame(const ImageFrame & myframe)
{

 outputVideo_ << myframe.color_;

  /*cv::namedWindow("mah", CV_WINDOW_AUTOSIZE);
  cv::imshow("mah", myframe.color_);

  short k = cvWaitKey(2);
  if (k == 27)
  {
      exit(-1);
  }
  */
 //convert depth in normal codec: from single channel 16 bit 
 //cv::Mat depthtosave;

 cv::Mat depthtosave;
 encodeDepth(myframe.depth_, depthtosave);

 timefile_ << std::to_string(myframe.time_stamp_.count()) << "\n";
}


void DepthExtractor::prepareOutputVideo(const std::string & path)
{
  //TODO: parse resolution from instance fields
  cv::Size S = cv::Size(cam_.getWidth(), cam_.getHeight());
  poseVideo_.open(path, CV_FOURCC('D','I','V','X'), 10, S, true);
  if (!poseVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }
}

void DepthExtractor::prepareVideo(const std::string & path)
{
  cv::Size S = cv::Size(cam_.getWidth(), cam_.getHeight());
  //cv::Size S = cv::Size(640, 480);
  outputVideo_.open(path, CV_FOURCC('D','I','V','X'), 30, S, true);
  if (!outputVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }
  std::string depthpath = "";

    if (depthpath_ == "")
    {
      depthpath = path + "depth.avi";
    }
    else
    {
      depthpath = depthpath_;
    }

    depthoutput_.open(depthpath,CV_FOURCC('B','G','R','A'), 30, S, true);

    if (!depthoutput_.isOpened())
    {
        std::cout  << "Could not open the depth output video for write: " << std::endl;
        exit(-1);
    }

  timefile_.open(path + "_timestamps.txt");
  jsonfile_.open(path + ".json"); 
}

void ONIDepthExtractor::prepareVideo(const std::string & path)
{
  cv::Size S = cv::Size(cam_.getHeight(), cam_.getWidth());

  outputVideo_.open(path, CV_FOURCC('D','I','V','X'), 30, S, true);
  if (!outputVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }
  std::string depthpath = "";

    if (depthpath_ == "")
    {
      depthpath = path + "depth.oni";
    }
    else
    {
      depthpath = depthpath_;
    }

    out_oni_.open(depthpath,std::ios::binary);

    if (!out_oni_.is_open())
    {
        std::cout  << "Could not open the depth output video for write: " << std::endl;
        exit(-1);
    }

  timefile_.open(path + "_timestamps.txt");
  jsonfile_.open(path + ".json"); 
}

size_t getFilesize(const char* filename) {
    struct stat st;
    stat(filename, &st);
    return st.st_size;
}

ONIDepthExtractor::ONIDepthExtractor(int argc, char**argv, DepthCamera & camera, const std::string & depth_video) : DepthExtractor(argc,argv,camera,depth_video)
{
  if(depth_video != "")
  {
    size_t filesize = getFilesize(depth_video.c_str());
    //Open file
    int fd = open(depth_video.c_str(), O_RDONLY, 0);
    if(fd == -1)
    {
      std::cout << "Error opening depth ONI file " << std::endl;
      exit(-1); 
    }
    //Execute mmap
    void* tmp = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

    if(in_oni_ == MAP_FAILED)
    {
      std::cout << "Error opening depth ONI file " << std::endl;
      exit(-1); 
    }

    in_oni_ = (uint8_t*)tmp;
  }
}

/**
* Converts depth into RGB image for MPEG encoding
*/
void ONIDepthExtractor::encodeDepth(const cv::Mat & depth, cv::Mat & output)
{ 

  cv::Mat depth_16(cv::Size(depth.rows,depth.cols),CV_16U);
  depth.convertTo(depth_16, CV_16U);


  std::vector<XnUInt8> depth_compressed(depth.cols * depth.rows * 2);
  XnUInt32 depth_8_size;

  //depth.convertTo(depth_16, CV_16U);
  XnStatus status = XnStreamCompressDepth16Z((const XnUInt16 *)depth_16.data,
                                   depth_compressed.size(), depth_compressed.data(),
                                   &depth_8_size);

  //std::cout << status << std::endl;

  out_oni_.write((const char *)&depth_8_size, 4);
  out_oni_.write((const char *)depth_compressed.data(), depth_8_size);
}

/**
* Decodes an encoded depth frame from kinect
*/
void ONIDepthExtractor::decodeDepth(const cv::Mat & rgb, cv::Mat & depth)
{ 
  //ignore rgb, use the binary file
  XnUInt32 nInputSize = 0;
  std::streampos size;

  XnUInt16 pOutput[rgb.cols*rgb.rows];
  XnUInt32 pnOutputSize = rgb.cols*rgb.rows*2;

  //TODO: allocate to_read data
  //use memcpy
  std::memcpy(&nInputSize, (&in_oni_[pos]), sizeof(XnUInt32));

  //in_oni_.read((unsigned char*)&nInputSize, sizeof(unsigned int));
  pos = pos + sizeof(XnUInt32);

  XnUInt8* pInput = (XnUInt8*)malloc(nInputSize);

  std::memcpy(pInput, (&in_oni_[pos]), nInputSize);
  pos = pos + nInputSize;
  //in_oni_.read(pInput,nInputSize);

  XnStatus status = XnStreamUncompressDepth16Z(pInput,nInputSize,pOutput,&pnOutputSize);

  //std::cout << rgb.cols << " " << rgb.rows << std::endl;
  //std::cout << status << std::endl;
  //exit(-1);

  //now must convert to cv::Mat
  depth = cv::Mat(rgb.rows,rgb.cols,CV_16UC1);
  depth.data = (unsigned char*)pOutput;

  cv::Mat depth_16(cv::Size(depth.rows,depth.cols),CV_32FC1);
  depth.convertTo(depth_16, CV_32FC1);

  depth = depth_16;

  /*cv::namedWindow("mah", CV_WINDOW_AUTOSIZE);
  cv::imshow("mah", depth);

  short k = cvWaitKey(2);
  if (k == 27)
  {
      exit(-1);
  }*/


}

void DepthExtractor::process(const std::string & write_keypoint, bool viz)
{ 

  PoseProcess(pose_params_, RGB_, poseKeypointsL_, outputImageL_);

  if(tracking2D_)
  {
    cv::cvtColor(RGB_, prev_gray_, CV_BGR2GRAY);
  }

  if( write_keypoint != "")
  {
    emitCSV(outputfile_, poseKeypointsL_, 0, cur_frame_);
  }

  if( viz)
  {
    visualize(&keep_on);
  }
}


void DepthExtractor::extract(const ImageFrame & m)
{ 

  RGB_ = m.color_;
  depth_ = m.depth_; 

  if(!live_)
  {
    cv::Mat tmp;
    decodeDepth(depth_, tmp);
    depth_ = tmp;
  }

  if(tracking2D_)
  {
    cv::cvtColor(RGB_, gray_, CV_BGR2GRAY);
  }

  cur_frame_ = cur_frame_ + skip_ + 1;
}

void ONIDepthExtractor::extract(const ImageFrame & m)
{ 

  RGB_ = m.color_;

  if(live_)
  {
    depth_ = m.depth_; 
  }
  else
  {
    cv::Mat tmp;
    decodeDepth(RGB_, tmp);
    depth_ = tmp;
  }

  if(tracking2D_)
  {
    cv::cvtColor(RGB_, gray_, CV_BGR2GRAY);
  }

  cur_frame_ = cur_frame_ + skip_ + 1;
}



void DepthExtractor::visualize(bool* keep_on)
{

  cv::namedWindow("Keypoints", CV_WINDOW_AUTOSIZE);
  cv::imshow("Keypoints", outputImageL_);

  short k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}

void DepthExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 
  cv::Mat verification = RGB_.clone();

  if(!pnts.empty())
  {

    std::vector<cv::Point2d> points2D(pnts.cols);

    cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_.intrinsics_,cv::Vec4d(0,0,0,0),points2D);

    for (auto & c : points2D)
    { 
      cv::circle(verification,c,6,cv::Scalar(255,0,0),5);
    }

    //TODO: draw also the keypoints got from the tracking
    if(tracked_)
    {
      for(int i = 0; i < trackedpnts_.cols; i++)
      {
        cv::Point3d pcc = trackedpnts_.at<cv::Point3d>(0,i);
        cv::Point2d pc =cv::Point2d(pcc.x, pcc.y);
        if(pcc.x != 0.0 && pcc.y != 0.0)
        {
          cv::circle(verification,pc,2,cv::Scalar(0,255,255),5);
        }
      }
    }

  }

  if(videooutput_)
  {
    poseVideo_ << verification;
  }

  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  cv::imshow("Verification", verification);
  
  short k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
  if (k == 's')
  {
    cv:imwrite("../data/3Dpoints.jpg", verification);
  }
}

double K2depthPoint(int col_x, int col_y, const cv::Mat & depth, const cv::Mat & RGB)
{
  //int d_x = depth.cols * col_x / RGB.cols;
  //int d_y = depth.rows * col_y / RGB.rows;

  float tmp = depth.at<float>(col_x, col_y);

  //be sure that the depth points are consistent
  //cv::circle(depth,cv::Point2d(d_x,d_y),3,cv::Scalar(255,255,255),5);
  //std::cout << tmp << std::endl;

  return (double)tmp; 
}

double Depth2Extractor::getDepthPoint(int x, int y)
{
  return K2depthPoint(x,y,depth_,RGB_); 
}

double ONIDepth2Extractor::getDepthPoint(int x, int y)
{
  return K2depthPoint(x,y,depth_,RGB_);
}

