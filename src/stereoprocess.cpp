#include "stereoprocess.h"
#include "opencv2/cudastereo.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>


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
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
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
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format.");

DEFINE_string(write_keypoint,           "",             "Full file path to write people body pose keypoints data. Only CSV format supported");  

DEFINE_bool(visualize,                  false,          "Visualize keypoints");

DEFINE_bool(show_error,                 false,           "Show the reprojection error on terminal");


PoseExtractor::PoseExtractor(int argc, char **argv, const std::string resolution)
{

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  inited_ = false;
  cur_frame_ = 0;

  if (FLAGS_write_keypoint != "")
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

  // Step 2 - Read Google flags (user defined configuration)
  // outputSize
  const auto outputSize = op::flagsToPoint(resolution, "1280x720");
  // netInputSize
  const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
  // netOutputSize
  const auto netOutputSize = netInputSize;
  // poseModel
  const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

  // Step 3 - Initialize all required classes
  cvMatToOpInput_ = new op::CvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};
  cvMatToOpOutput_ = new op::CvMatToOpOutput{outputSize};
  poseExtractorCaffeL_ = new op::PoseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,
                                                FLAGS_model_folder, FLAGS_num_gpu_start};
  poseRendererL_ = new op::PoseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_render_threshold,
                                    !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
  opOutputToCvMatL_ = new op::OpOutputToCvMat{outputSize};
  opOutputToCvMatR_ = new op::OpOutputToCvMat{outputSize};
}

double PoseExtractor::go(const cv::Mat & image, const bool ver, cv::Mat & points3D, bool* keep_on)
{ 

  extract(image);

  process();
 
  double error = triangulate(points3D);

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
    poseExtractorCaffeL_->initializationOnThread();
    poseRendererL_->initializationOnThread();
    inited_ = true;
  }
}

void PoseExtractor::destroy()
{
  outputfile_.close();
}

void PoseExtractor::setDepth(const cv::Mat & m)
{
  depth_ = m;
}


StereoPoseExtractor::StereoPoseExtractor(int argc, char **argv, const std::string resolution) : PoseExtractor(argc, argv, resolution), 
                                            cam_(resolution)
{  

  if (FLAGS_write_video != "")
  { 
    //TODO: parse resolution from instance fields
    cv::Size S = cv::Size(cam_.width_*2, cam_.height_);
    outputVideo_.open(FLAGS_write_video, CV_FOURCC('M','J','P','G'), 7, S, true);
    if (!outputVideo_.isOpened())
    {
        std::cout  << "Could not open the output video for write: " << std::endl;
        exit(-1);
    }
  }

}

void StereoPoseExtractor::triangulateCore(cv::Mat & cam0pnts, cv::Mat & cam1pnts, cv::Mat & finalpoints)
{
  int N = 0;
  cv::Mat nz_cam0pnts;
  cv::Mat nz_cam1pnts;
  //TODO: check the numeber of detected people is the same, otherwise PROBLEM

  if (cam0pnts.cols == 0 || cam1pnts.cols == 0)
  {
    std::cout << "One Image did not get points " << std::endl;
    return;
  }

  if(cam0pnts.cols != cam1pnts.cols)
  {
    std::cout << "number of detectd people differs" << std::endl;
    std::cout << cam0pnts.cols << " " << cam1pnts.cols << std::endl;
    //TODO: routine to take only the bounding boxes of the same people 
    equalize(cam0pnts, cam1pnts, cam0pnts, cam1pnts);
  }

  N = nz_cam1pnts.cols;

  //Undistort points
  cv::Mat cam0pnts_undist(1,N,CV_64FC2);
  cv::Mat cam1pnts_undist(1,N,CV_64FC2);

  //If zeros in at least one: remove both 
  filterVisible(cam0pnts, cam1pnts, cam0pnts, cam1pnts);

  if(cam0pnts.cols == 0)
  {
    std::cout <<  "NO MATCHING POINTS FOUND!" << std::endl; 
    return;
  }

  N = cam0pnts.cols;
  cv::Mat pnts3d(1,N,CV_64FC4);

  cv::Mat R1,R2,P1,P2,Q;
  /*Computes rectification transforms for each head of a calibrated stereo camera*/
  //cv::stereoRectify(cam_.intrinsics_left_, cam_.dist_left_, cam_.intrinsics_right_, cam_.dist_right_, cv::Size(cam_.width_,cam_.height_), cam_.SR_,cam_.ST_, R1, R2, P1, P2, Q);

  cv::undistortPoints(cam0pnts, cam0pnts_undist, cam_.intrinsics_left_, cam_.dist_left_, cam_.intrinsics_left_);
  cv::undistortPoints(cam1pnts, cam1pnts_undist, cam_.intrinsics_right_, cam_.dist_right_, cam_.intrinsics_right_);

  cv::Mat proj_left,proj_right;

  proj_left = cv::Mat(3,4,CV_64FC1);
  proj_right = cv::Mat(3,4,CV_64FC1);

  for (int i = 0; i < 3; i++)
  {
    for ( int j = 0; j < 3; j++)
    {
      proj_left.at<double>(i,j) = cam_.intrinsics_left_.at<double>(i,j);
    }
  }

  for ( int i = 0; i < 3; i++)
  {
    proj_left.at<double>(i,3) = 0.0;
  }


  cv::Mat rototran;
  cv::hconcat(cv::Mat::eye(3,3,CV_64FC1), cam_.ST_, rototran);

  proj_right = cam_.intrinsics_left_ * rototran;

  cv::triangulatePoints(proj_left, proj_right, cam0pnts_undist, cam1pnts_undist, pnts3d);

  finalpoints = cv::Mat(1,N,CV_64FC3);

  for (int i = 0; i < N; i++)
  { 
    cv::Vec4d cur = pnts3d.col(i);
    cv::Vec3d p3d(cur[0]/cur[3], cur[1]/cur[3],cur[2]/cur[3]);
    finalpoints.at<cv::Vec3d>(0,i) = p3d;
  }
}


void StereoPoseExtractor::extract(const cv::Mat & image)
{

  cur_frame_ ++;
  splitVertically(image, imageleft_, imageright_);

}

//TODO: save time by using OpenPose in a single image? 
void StereoPoseExtractor::process()
{

  op::Array<float> netInputArrayL;
  op::Array<float> netInputArrayR;

  op::Array<float> outputArrayL;
  op::Array<float> outputArrayR;

  std::vector<float> scaleRatiosL;
  std::vector<float> scaleRatiosR;

  double scaleInputToOutputL;
  double scaleInputToOutputR;

  std::tie(netInputArrayL, scaleRatiosL) = cvMatToOpInput_->format(imageleft_);
  std::tie(scaleInputToOutputL, outputArrayL) = cvMatToOpOutput_->format(imageleft_);
  std::tie(netInputArrayR, scaleRatiosR) = cvMatToOpInput_->format(imageright_);
  std::tie(scaleInputToOutputR, outputArrayR) = cvMatToOpOutput_->format(imageright_);

  // Step 3 - Estimate poseKeypoints
  poseExtractorCaffeL_->forwardPass(netInputArrayL, {imageleft_.cols, imageleft_.rows}, scaleRatiosL);
  poseKeypointsL_ = poseExtractorCaffeL_->getPoseKeypoints();

  poseExtractorCaffeL_->forwardPass(netInputArrayR, {imageright_.cols, imageright_.rows}, scaleRatiosR);
  poseKeypointsR_ = poseExtractorCaffeL_->getPoseKeypoints();

  std::string kpl_str = poseKeypointsL_.toString();
  std::string kpr_str = poseKeypointsR_.toString();

  // Step 4 - Render poseKeypoints
  poseRendererL_->renderPose(outputArrayL, poseKeypointsL_);
  poseRendererL_->renderPose(outputArrayR, poseKeypointsR_);    
  
  // Step 5 - OpenPose output format to cv::Mat
  outputImageL_ = opOutputToCvMatL_->formatToCvMat(outputArrayL);
  outputImageR_ = opOutputToCvMatL_->formatToCvMat(outputArrayR);

  if( FLAGS_write_video != "")
  { 
    cv::Mat sidebyside_in;
    cv::hconcat(imageleft_, imageright_, sidebyside_in);
    outputVideo_ << sidebyside_in;
  }

  if( FLAGS_write_keypoint != "")
  {
    emitCSV(outputfile_, kpl_str, poseKeypointsL_, 0, cur_frame_);
    emitCSV(outputfile_, kpr_str, poseKeypointsR_, 1, cur_frame_);
  }

  if( FLAGS_visualize)
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
*   
*/
double StereoPoseExtractor::triangulate(cv::Mat & finalpoints)
{ 

  //I can take all the points negleting if they belong to a specific person 
  //how can I know if the points belong to the same person? 
  cv::Mat cam0pnts;
  cv::Mat cam1pnts;

  getPoints(cam0pnts,cam1pnts);

  triangulateCore(cam0pnts, cam1pnts, finalpoints);

  return getRMS(cam0pnts,finalpoints);

}

void StereoPoseExtractor::visualize(bool * keep_on)
{
  //TODO: make a video with 2 frame side by side
  cv::Mat sidebyside_out;
  cv::hconcat(outputImageL_, outputImageR_, sidebyside_out);

  cv::namedWindow("Side By Side", CV_WINDOW_AUTOSIZE);
  cv::imshow("Side By Side", sidebyside_out);

  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}


void StereoPoseExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{ 

  if(pnts.empty())
  {
    return;
  }
  

  std::vector<cv::Point2d> points2D(pnts.cols);

  cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(cam_.ST_[0],0,0),cam_.intrinsics_right_,cam_.dist_right_,points2D);

  int inside = 0;

  for (unsigned int i = 0; i < pnts.cols; i++)
  {

    if(points2D[i].x < cam_.width_ && points2D[i].y < cam_.height_&& points2D[i].x > 0 && points2D[i].y > 0)
    {
      inside ++;
    }
  } 
  //TODO: write circles in projected points
  cv::Mat verification = imageright_.clone();
  for (auto & c : points2D)
  {
    cv::circle(verification,c,4,cv::Scalar(0,0,255),2);
  }


  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  cv::imshow("Verification", verification);
  
  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
  if (k == 's')
  {
    cv:imwrite("../data/3Dpoints.jpg", verification);
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
    cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),cam_.intrinsics_left_,cam_.dist_left_,points2D);
  }
  else
  {
    cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(cam_.ST_[0],0,0),cam_.intrinsics_right_,cam_.dist_right_,points2D);
  }


  cv::transpose(points2D,points2D);

  return cv::norm(points2D - cam0pnts);
}


PoseExtractorFromFile::PoseExtractorFromFile(int argc, char **argv, const std::string resolution, const std::string path) 
                                              : StereoPoseExtractor(argc,argv,resolution), filepath_(path), file_(path)
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

DepthExtractor::DepthExtractor(int argc, char **argv, const std::string resolution) : PoseExtractor(argc, argv, resolution)
{

  // Step 2 - Read Google flags (user defined configuration)
  // outputSize
  const auto outputSize = op::flagsToPoint("640x480", "640x480");
  // netInputSize
  const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "640x480");
  // netOutputSize
  const auto netOutputSize = netInputSize;
  // poseModel
  const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

  // Step 3 - Initialize all required classes
  cvMatToOpInput_ = new op::CvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};
  cvMatToOpOutput_ = new op::CvMatToOpOutput{outputSize};
  poseExtractorCaffeL_ = new op::PoseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,
                                                FLAGS_model_folder, FLAGS_num_gpu_start};
  poseRendererL_ = new op::PoseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_render_threshold,
                                    !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
  opOutputToCvMatL_ = new op::OpOutputToCvMat{outputSize};
  opOutputToCvMatR_ = new op::OpOutputToCvMat{outputSize};

  pcam_ = new DepthCamera();
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

  double fx = pcam_->intrinsics_.at<double>(0,0);
  double fy = pcam_->intrinsics_.at<double>(1,1);
  double cx = pcam_->intrinsics_.at<double>(0,2);
  double cy = pcam_->intrinsics_.at<double>(1,2);

  double Z = z;
  double X = ((u - cx) * Z)/fx;
  double Y = ((v - cy) * Z)/fy;

  return cv::Point3d(X,Y,Z);

}

double DepthExtractor::getRMS(const cv::Mat & cam0pnts, const cv::Mat & pnts3D)
{
  if(pnts3D.empty())
  {
    return 0.0;
  }

  cv::Mat points2D; 

  cv::projectPoints(pnts3D,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),pcam_->intrinsics_,cv::Vec4d(0,0,0,0),points2D);

  cv::transpose(points2D,points2D);

  return cv::norm(points2D - cam0pnts);
}

double DepthExtractor::triangulate(cv::Mat & finalpoints)
{ 
  //I can take all the points negleting if they belong to a specific person 
  //how can I know if the points belong to the same person? 
  cv::Mat cam0pnts;
  opArray2Mat(poseKeypointsL_, cam0pnts);

  std::vector<cv::Point3d> points3D;
  std::vector<cv::Point2d> points2D;

  //If zeros in at least one: remove both 
  filterVisible(cam0pnts, cam0pnts);

  for( int i = 0; i < cam0pnts.cols; i++)
  { 
    cv::Point2d keypoint = cam0pnts.at<cv::Point2d>(0,i);
    cv::Point3d point = getPointFromDepth(keypoint.x,keypoint.y,
                        (double)depth_.at<uint16_t>(cvRound(keypoint.x),cvRound(keypoint.y)));

    point = point / 1000;

    if(point != cv::Point3d(0,0,0))
    {
      points3D.push_back(point);
      points2D.push_back(keypoint);
    }
  }

  //TODO: remove zeros also from points3D and the correspondent from points 2D
  cam0pnts = cv::Mat(points2D);
  cv::transpose(cam0pnts, cam0pnts);

  std::cout << "Nose: " << points3D[0] << std::endl;

  finalpoints = cv::Mat(points3D);

  return getRMS(cam0pnts,finalpoints);

  return 0.0;
}

void DepthExtractor::process()
{

  op::Array<float> netInputArrayL;

  op::Array<float> outputArrayL;

  std::vector<float> scaleRatiosL;

  double scaleInputToOutputL;

  std::tie(netInputArrayL, scaleRatiosL) = cvMatToOpInput_->format(RGB_);
  std::tie(scaleInputToOutputL, outputArrayL) = cvMatToOpOutput_->format(RGB_);

  // Step 3 - Estimate poseKeypoints
  poseExtractorCaffeL_->forwardPass(netInputArrayL, {RGB_.cols, RGB_.rows}, scaleRatiosL);

  poseKeypointsL_ = poseExtractorCaffeL_->getPoseKeypoints();

  std::string kpl_str = poseKeypointsL_.toString();

  // Step 4 - Render poseKeypoints
  poseRendererL_->renderPose(outputArrayL, poseKeypointsL_);
  
  // Step 5 - OpenPose output format to cv::Mat
  outputImageL_ = opOutputToCvMatL_->formatToCvMat(outputArrayL);

  if( FLAGS_write_video != "")
  { 
    outputVideo_ << RGB_;
  }

  if( FLAGS_write_keypoint != "")
  {
    emitCSV(outputfile_, kpl_str, poseKeypointsL_, 0, cur_frame_);
  }

  if( FLAGS_visualize)
  {
    visualize(&keep_on);
  }
}

void DepthExtractor::extract(const cv::Mat & m)
{  
  cur_frame_ ++;
  RGB_ = m;
}

void DepthExtractor::visualize(bool* keep_on)
{

 
  cv::namedWindow("Keypoints", CV_WINDOW_AUTOSIZE);
  cv::imshow("Keypoints", outputImageL_);

  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
}

void DepthExtractor::verify(const cv::Mat & pnts, bool* keep_on)
{
  if(pnts.empty())
  {
    return;
  }

  std::vector<cv::Point2d> points2D(pnts.cols);

  cv::projectPoints(pnts,cv::Mat::eye(3,3,CV_64FC1),cv::Vec3d(0,0,0),pcam_->intrinsics_,cv::Vec4d(0,0,0,0),points2D);

  cv::Mat verification = RGB_.clone();
  for (auto & c : points2D)
  {
    cv::circle(verification,c,4,cv::Scalar(0,0,255),2);
  }


  cv::namedWindow("Verification", CV_WINDOW_AUTOSIZE);
  cv::imshow("Verification", verification);
  
  int k = cvWaitKey(2);
  if (k == 27)
  {
      *keep_on = false;
  }
  if (k == 's')
  {
    cv:imwrite("../data/3Dpoints.jpg", verification);
  }
}
