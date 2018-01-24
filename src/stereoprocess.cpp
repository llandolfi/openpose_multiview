#include "stereoprocess.h"
#include "opencv2/cudastereo.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>
#include <boost/system/error_code.hpp>
#include <json/json.h>
// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "/home/lando/projects/openpose/models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "640x480",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");

DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results on a black background.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");

DEFINE_string(write_keypoint,           "",             "Full file path to write people body pose keypoints data. Only CSV format supported");  

DEFINE_string(write_keypoint3D,         "",             "Full file path to write people body pose keypoints 3D. Only CSV format supported");  

DEFINE_bool(visualize,                  false,          "Visualize keypoints");

DEFINE_bool(show_error,                 false,           "Show the reprojection error on terminal");

DEFINE_int32(udp_port,                  9999,               "Stream body data points in JSON format to defined port");

DEFINE_string(udp_address,              "127.0.0.1",      "Stream body data points in JSON format to defined port");

DEFINE_bool(2Dtracking,                  false,         "Track 2D points points instead of detecting on each frame");


PoseExtractor::PoseExtractor(int argc, char **argv, Camera & camera) : udpstreamer_(FLAGS_udp_port, FLAGS_udp_address)
{

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  inited_ = false;
  cur_frame_ = 0;

  if(FLAGS_write_keypoint != "")
  {
    outputfile_.open(FLAGS_write_keypoint);
    //TODO:write header of outputfilef
    outputfile_ << "camera frame subject ";
    for (int i = 0; i < 18; i++)
    {
      outputfile_ << "p" << i << "x" << " p" << i << "y" << " p" << i << "conf ";
    }
    outputfile_ << "\n";
  }

  if(FLAGS_write_keypoint3D != "")
  {
    outputfile3D_.open(FLAGS_write_keypoint3D);
    //TODO:write header of outputfilef
    outputfile3D_ << "camera frame subject ";
    for (int i = 0; i < 18; i++)
    {
      outputfile3D_ << "p" << i << "x" << " p" << i << "y" << " p" << i << "z ";
    }
    outputfile3D_ << "\n";
  }

  if(FLAGS_2Dtracking)
  {
    tracking2D_ = true;
  }

  const bool enableGoogleLogging = true;
  // Step 2 - Read Google flags (user defined configuration)
  // outputSize
  const auto outputSize = op::flagsToPoint(camera.getResolution(), "1280x720");
  // netInputSize
  const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "640x480");
  // netOutputSize
  const auto netOutputSize = netInputSize;
  // poseModel
  const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

  pose_params_.scaleAndSizeExtractor_ = new op::ScaleAndSizeExtractor (netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);

  pose_params_.poseExtractorCaffe_ = new op::PoseExtractorCaffe{poseModel, FLAGS_model_folder,
                                        FLAGS_num_gpu_start, {}, op::ScaleMode::ZeroToOne, enableGoogleLogging};
  pose_params_.poseRenderer_ = new op::PoseCpuRenderer {poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                      (float)FLAGS_alpha_pose};
  pose_params_.frameDisplayer_ = new op::FrameDisplayer{"OpenPose multiview", outputSize};
}


std::string StereoPoseExtractor::pnts2JSON(const cv::Mat & pnts, int frame, const std::string & time)
{

  Json::Value points;
  Json::Value colors;

  cam_->JSONPoints(pnts,points);
  
  colors["r"] = 0;
  colors["g"] = 0;
  colors["b"] = 1.0;
  

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

double PoseExtractor::go(const ImageFrame & image, const bool ver, cv::Mat & points3D, bool* keep_on)
{ 

  points3D = cv::Mat();
  double error = 0.0; 
  bool tracked = false; 

  extract(image);

  //Trackable if previous frame was detection or no error in previous tracking
  if(tracking2D_)
  {  
    tracked_ = track();
  }

  if(!tracked_)
  { 
    process(FLAGS_write_keypoint, FLAGS_visualize);
  }

  error = triangulate(points3D);

  if(FLAGS_udp_port != 0)
  {
    //TODO: generate JSON message, send with updstreamer
    udpstreamer_.sendMessage(pnts2JSON(points3D, cur_frame_, std::to_string(image.time_stamp_.count())));
  }
  //TODO: add specific flag (output format)
  else
  {
    jsonfile_ << pnts2JSON(points3D, cur_frame_, std::to_string(image.time_stamp_.count()));
  }

  if(FLAGS_write_keypoint3D != "")
  {
    emitCSV3D(outputfile3D_, poseKeypointsL_, cur_frame_, points3D);
  }


  if( FLAGS_show_error)
  {
    std::cout << "Reprojection error: " << error << std::endl;
  }

  if(ver)
  { 
    verify(points3D, keep_on);
  }

  return error;
}

void PoseExtractor::init()
{
  if (inited_ == false)
  {
    pose_params_.poseExtractorCaffe_->initializationOnThread();
    pose_params_.poseRenderer_->initializationOnThread();
    inited_ = true;
  }
}

void PoseExtractor::destroy()
{
  outputfile_.close();
  outputfile3D_.close();
  timefile_.close();
}

void PoseExtractor::setDepth(const cv::Mat & m)
{
  depth_ = m;
}


StereoPoseExtractor::StereoPoseExtractor(int argc, char **argv, StereoCamera & cam) : PoseExtractor(argc, argv, cam)                              
{  
  cam_ = &(cam);
}


void StereoPoseExtractor::prepareVideo(const std::string & path)
{
  //TODO: parse resolution from instance fields
  cv::Size S = cv::Size(cam_->width_*2, cam_->height_);
  outputVideo_.open(path, CV_FOURCC('D','I','V','X'), cam_->fps_, S, true);
  if (!outputVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }

  timefile_.open(path + "_timestamps.txt");
  jsonfile_.open(path + ".json"); 
}

void StereoPoseExtractor::prepareOutputVideo(const std::string & path)
{
  //TODO: parse resolution from instance fields
  cv::Size S = cv::Size(cam_->width_, cam_->height_);
  poseVideo_.open(path, CV_FOURCC('D','I','V','X'), 10, S, true);
  if (!poseVideo_.isOpened())
  {
      std::cout  << "Could not open the output video for write: " << std::endl;
      exit(-1);
  }
}


void filterUncertain(const double thresh, cv::Mat & input)
{ 

  std::vector<cv::Vec3d> v;

  for(int i = 0; i < input.cols; i++)
  {
    cv::Vec3d point = input.at<cv::Vec3d>(0,i);
    if(point[2] < thresh)
    {
      input.at<cv::Vec3d>(0,i) = cv::Vec3d(0.0,0.0,0.0);
    }
  }

}

//N.B. cam0pnts holds also confidence
void StereoPoseExtractor::triangulateCore(cv::Mat & cam0pnts, cv::Mat & cam1pnts, cv::Mat & finalpoints)
{ 

  int N = 0;
  cv::Mat cam0pnts_undist;
  cv::Mat cam1pnts_undist;

  if (cam0pnts.cols == 0 || cam1pnts.cols == 0)
  {
    std::cout << "One Image did not get points. No correspondences can be found!" << std::endl;
    return;
  }

  //remve the points with confidence less yhan a threshold
  filterUncertain(0.40, cam0pnts);
  filterUncertain(0.40, cam1pnts);

  std::map<int,int> correspondences;
  findCorrespondences(cam0pnts, cam1pnts, cam0pnts, cam1pnts);

  //TODO: check not emptyness
  if(cam0pnts.empty() || cam1pnts.empty())
  {
    std::cout << "No correspondences " << std::endl;
    return;
  }

  N = cam0pnts.cols;
  cv::Mat pnts3d(1,N,CV_64FC4);

  cv::Mat R1,R2,P1,P2,Q;
  /*Computes rectification transforms for each head of a calibrated stereo camera*/
  //cv::stereoRectify(cam_->camera_left_.intrinsics_, cam_->camera_left.dist_, cam_->camera_right_intrinsics_, cam_->camera_right_.dist_, cv::Size(cam_->width_,cam_->height_), cam_->SR_,cam_->ST_, R1, R2, P1, P2, Q);

  cv::undistortPoints(cam0pnts, cam0pnts_undist, cam_->camera_left_.intrinsics_, cam_->camera_left_.dist_, cam_->camera_left_.intrinsics_);
  cv::undistortPoints(cam1pnts, cam1pnts_undist, cam_->camera_right_.intrinsics_, cam_->camera_right_.dist_, cam_->camera_right_.intrinsics_);

  cv::Mat proj_left,proj_right;

  proj_left = cv::Mat(3,4,CV_64FC1);
  proj_right = cv::Mat(3,4,CV_64FC1);

  for (int i = 0; i < 3; i++)
  {
    for ( int j = 0; j < 3; j++)
    {
      proj_left.at<double>(i,j) = cam_->camera_left_.intrinsics_.at<double>(i,j);
    }
  }

  for ( int i = 0; i < 3; i++)
  {
    proj_left.at<double>(i,3) = 0.0;
  }


  cv::Mat rototran;
  cv::hconcat(cam_->SR_, cam_->ST_, rototran);

  proj_right = cam_->camera_left_.intrinsics_ * rototran;

  cv::triangulatePoints(proj_left, proj_right, cam0pnts_undist, cam1pnts_undist, pnts3d);

  finalpoints = cv::Mat(1,N,CV_64FC3);

  for (int i = 0; i < N; i++)
  { 
    cv::Vec4d cur = pnts3d.col(i);

    //TODO: check for negative Z and investigate
    cv::Vec3d p3d(cur[0]/cur[3], cur[1]/cur[3],cur[2]/cur[3]);

    if(false)
    {
      std::cout << "Negative Z in " << i << std::endl;
      double n = std::numeric_limits<double>::quiet_NaN();
      finalpoints.at<cv::Vec3d>(0,i) = cv::Vec3d(n,n,n);
    }
    else
    {
    finalpoints.at<cv::Vec3d>(0,i) = p3d;
    }

  }
}

bool StereoPoseExtractor::track()
{
  if(prev_gray_.empty() || prev_grayR_.empty())
  {
    return false;
  }

  if(cur_frame_ % 5 == 0)
  {
    points_[0].clear();
    pointsR_[0].clear();
    return false;
  }

  if(points_[0].size() == 0)
  {
   cv::Mat bodypartsL;
   opArray2Mat(poseKeypointsL_, bodypartsL);
   mat2Vector(bodypartsL,points_[0]);
  }

  if(pointsR_[0].size() == 0)
  {
   cv::Mat bodypartsR;
   opArray2Mat(poseKeypointsR_, bodypartsR);
   mat2Vector(bodypartsR,pointsR_[0]);
  }

  tracked_left_ = trackLK(prev_gray_, gray_, points_[0], points_[1], 2.0, trackedpnts_);
  tracked_right_ = trackLK(prev_grayR_, grayR_, pointsR_[0], pointsR_[1], 2.0, trackedpntsR_);

  if(!tracked_left_)
  {
    points_[0].clear();
  }
  else
  {
    prev_gray_ = gray_.clone();
  }

  if(!tracked_right_)
  {
    pointsR_[0].clear();
  }
  else
  {
    //Try to swap?
    prev_grayR_ = grayR_.clone();
  }

  return tracked_left_ && tracked_right_;
}


void StereoPoseExtractor::extract(const ImageFrame & image)
{

  cur_frame_ ++;
  splitVertically(image.color_, imageleft_, imageright_);

  if(tracking2D_)
  {
    cv::cvtColor(imageleft_, gray_, CV_BGR2GRAY);
    cv::cvtColor(imageright_, grayR_, CV_BGR2GRAY);
  }

}

void PoseExtractor::process(const std::string & write_keypoint, bool viz)
{
  PoseProcess(pose_params_, imageleft_, poseKeypointsL_, outputImageL_);
}


void StereoPoseExtractor::appendFrame(const ImageFrame & myframe)
{ 
  outputVideo_ << myframe.color_;
  timefile_ << std::to_string(myframe.time_stamp_.count()) << "\n";
}

void StereoPoseExtractor::process(const std::string & write_keypoint, bool viz)
{ 


  if(!tracked_left_ && !tracked_right_)
  {
    std::cout << "ALL detected " << std::endl;
  }

  //TODO: check if tracked left or right, act consequently
  if(!tracked_left_)
  {
    PoseExtractor::process(write_keypoint, viz);
  }
  else
  {
    std::cout << "track left " << std::endl;
  }
  
  if(!tracked_right_)
  {
    PoseProcess(pose_params_, imageright_, poseKeypointsR_, outputImageR_);
  }
  else
  {
    std::cout << "track right " << std::endl;
  }

  if(tracking2D_)
  { 
    if(!tracked_left_)
    {
      cv::cvtColor(imageleft_,prev_gray_,CV_BGR2GRAY);
    }
    if(!tracked_right_)
    {
      cv::cvtColor(imageright_,prev_grayR_,CV_BGR2GRAY);
    }
  }

  if( write_keypoint != "")
  {
    emitCSV(outputfile_, poseKeypointsL_, 0, cur_frame_);
    emitCSV(outputfile_, poseKeypointsR_, 1, cur_frame_);
  }

  if( viz)
  {
    visualize(&keep_on);
  }
}

void StereoPoseExtractor::getPoints(cv::Mat & outputL, cv::Mat & outputR)
{
  opArray2Mat(poseKeypointsL_, outputL);
  opArray2Mat(poseKeypointsR_, outputR);
}

/*
* Returns the 3D points from stereo couples acquired from stereo camera  
*/
double StereoPoseExtractor::triangulate(cv::Mat & finalpoints)
{ 

  //I can take all the points negleting if they belong to a specific person 
  //how can I know if the points belong to the same person? 
  cv::Mat cam0pnts;
  cv::Mat cam1pnts;

  if(!tracked_left_)
  {
    opArray2Mat(poseKeypointsL_, cam0pnts); 
  }
  else
  {
    cam0pnts = trackedpnts_;
  }

  if(!tracked_right_)
  {
    opArray2Mat(poseKeypointsR_, cam1pnts);
  }
  else
  {
    cam1pnts = trackedpntsR_;
  }


  if(cam0pnts.empty() || cam1pnts.empty())
  {
    return 0.0;
  }

  triangulateCore(cam0pnts, cam1pnts, finalpoints);

  return getRMS(cam0pnts,finalpoints);

}

void StereoPoseExtractor::visualize(bool * keep_on)
{ 

  cv::Mat cam0pnts, cam1pnts;
  getPoints(cam0pnts,cam1pnts);

  //TODO: draw circles of different colors depending on body index 
  for(int i = 0; i < cam0pnts.cols/18; i++)
  { 
    //std::cout << "i: " << i << std::endl;
    for (int j = 0; j < 18; j++)
    {
    cv::Point3d pc = cam0pnts.at<cv::Point3d>(0,j + (i*18));
    cv::Point2d p(pc.x, pc.y);
    cv::putText(outputImageL_, std::to_string(i), p,  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250),1,CV_AA);
    //cv::Scalar color = cv::Scalar(255 - (i*75),0,i*70);
    //cv::circle(outputImageL_,p,6,color,5);
    }
  }

  for(int i = 0; i < cam1pnts.cols/18; i++)
  {
    for (int j = 0; j < 18; j++)
    {
    cv::Point3d pc = cam1pnts.at<cv::Point3d>(0,j + (i*18));
    cv::Point2d p(pc.x, pc.y);
    cv::putText(outputImageR_, std::to_string(i), p, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250),1,CV_AA);
    //cv::Scalar color = cv::Scalar(255 - (i*75),0,i*70);
    //cv::circle(outputImageR_,p,6,color,5);
    }
  }



  //TODO: make a video with 2 frame side by side
  cv::Mat sidebyside_out;
  cv::hconcat(outputImageL_, outputImageR_, sidebyside_out);


  cv::namedWindow("Side By Side", CV_WINDOW_AUTOSIZE);
  cv::imshow("Side By Side", sidebyside_out);

  short k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}


void StereoPoseExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 

  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);

  cv::Mat verification = imageright_.clone();

  if(!pnts.empty())
  {

    std::vector<cv::Point2d> points2D(pnts.cols);

    cv::projectPoints(pnts,cam_->SR_,cam_->ST_,cam_->camera_right_.intrinsics_,cam_->camera_right_.dist_,points2D);

    int inside = 0;

    for (unsigned int i = 0; i < pnts.cols; i++)
    {

      if(points2D[i].x < cam_->width_ && points2D[i].y < cam_->height_&& points2D[i].x > 0 && points2D[i].y > 0)
      {
        inside ++;
      }
    } 

    for (int i = 0; i < pnts.cols; i++)
    {
      if(pnts.at<cv::Vec3d>(0,i)[2] < 99999)
      {
        cv::circle(verification,points2D[i],6,cv::Scalar(255,0,0),5);
      }
    }

    if(tracked_right_)
    {
      for(int i = 0; i < trackedpntsR_.cols; i++)
      {
        cv::Point3d pcc = trackedpntsR_.at<cv::Point3d>(0,i);
        cv::Point2d pc = cv::Point2d(pcc.x, pcc.y);
        if(pcc.x != 0.0 && pcc.y != 0.0)
        {
          cv::circle(verification,pc,2,cv::Scalar(0,255,255),5);
        }
      }
    }

  } 

  if(videooutput_)
  {
    std::cout << "writing video " << std::endl;
    poseVideo_ << verification;
  }

  cv::imshow("Verification", verification);
  

  short k = cvWaitKey(3);

  if (k == 27)
  {   
      *keep_on = false;
  }
  if (k == 's')
  { 
    std::cout << "SAVING " << std::endl;
    cv::imwrite("../data/3Dpoints.jpg", verification);
  }
}

double StereoPoseExtractor::getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D, bool left)
{ 

  if(pnts3D.empty())
  {
    return 0.0;
  }

  cv::Mat points2D; 

  if(left)
  {
    cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_->camera_left_.intrinsics_,cam_->camera_left_.dist_,points2D);
  }
  else
  {
    cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(cam_->ST_[0],0,0),cam_->camera_right_.intrinsics_,cam_->camera_right_.dist_,points2D);
  }


  cv::transpose(points2D,points2D);

  return cv::norm(points2D - cam0pnts);
}


PoseExtractorFromFile::PoseExtractorFromFile(int argc, char **argv, StereoCamera & camera, const std::string path) 
                                              : StereoPoseExtractor(argc,argv,camera), filepath_(path), file_(path)
{ 


  if(file_.is_open())
  {
    getline(file_,line_);
    getline(file_,line_);
  }
  else
  {
    std::cout << "Could not open keypoints file!" << std::endl;
    exit(-1);
  } 
}


/*
* Fill vector lines with the file rows relative to current frame
*/
void PoseExtractorFromFile::getNextBlock(std::vector<std::vector<std::string>> & lines)
{

  bool keep = true;

  while(keep)
  {
    std::vector<std::string> tokens = CSVTokenize(line_);

    if(atoi(tokens[1].c_str()) == cur_frame_)
    {
      lines.push_back(tokens);
      getline(file_,line_); 
    }
    else{
      keep = false;
    }
  }
}


void fillPointsFromFile(const std::vector<std::string> & line, std::vector<cv::Point2d> & points)
{

  for(int i = 3; i < line.size(); i = i + 3)
  {
    cv::Point2d point(atof(line[i].c_str()), atof(line[i+1].c_str()));
    points.push_back(point);
  }
}

void PoseExtractorFromFile::process(const cv::Mat & image)
{ 

  cur_frame_ ++;

  splitVertically(image, imageleft_, imageright_);

  std::vector<std::vector<std::string>> frametokens;

  points_left_.clear();
  points_right_.clear();

  getNextBlock(frametokens);

  for (auto & s : frametokens)
  {
    if (strcmp(s[0].c_str(), "0") == 0)
    {
      fillPointsFromFile(s,points_left_);
    }
    else
    {
      fillPointsFromFile(s,points_right_);
    }
  }
}

void PoseExtractorFromFile::visualize(bool * keep_on)
{ 

  cv::Mat sidebyside_out;

  outputImageL_ = imageleft_.clone();
  outputImageR_ = imageright_.clone();

  for (auto & c : points_left_)
  {
    cv::circle(outputImageL_,c,4,cv::Scalar(0,0,255),2);
  }

  for (auto & c : points_right_)
  {
    cv::circle(outputImageR_,c,4,cv::Scalar(0,0,255),2);
  }

  cv::hconcat(outputImageL_, outputImageR_, sidebyside_out);

  cv::namedWindow("Side By Side", CV_WINDOW_AUTOSIZE);
  cv::imshow("Side By Side", sidebyside_out);

  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}


void PoseExtractorFromFile::getPoints(cv::Mat & outputL, cv::Mat & outputR)
{
  vector2Mat(points_left_, outputL);
  vector2Mat(points_right_, outputR);
}


