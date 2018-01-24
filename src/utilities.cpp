#include "utilities.hpp"
#include "opencv2/video/tracking.hpp"
#include <limits>
#include <map>

/*
 *  Returns height from resolution string, which is in the form widthxheight
*/
int getHeight(const std::string & resolution)
{
  std::string height = resolution.substr(resolution.find("x") + 1);
  return atoi(height.c_str());
}
/*
* Returns width from resolution string, which is in the form widthxheight
*/
int getWidth(const std::string & resolution)
{
  std::string width = resolution.substr(0,resolution.find("x"));
  return atoi(width.c_str());
}

int getInt(const std::string & s, const std::string c)
{
	std::string height = s.substr(s.find(c) + 1);
	return atoi(height.c_str());	
}

double getDouble(const std::string & s, const std::string c)
{
	std::string height = s.substr(s.find(c) + 1);
	return atof(height.c_str());	
}

//*Black Magic*/
constexpr unsigned int str2int(const char* str, int h)
{
  return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

/*
* Associates resolution code to resolution string
*/
const std::string getResolutionCode(const std::string resolution)
{
  switch(str2int(resolution.c_str()))
  {
    case str2int("672x376"):
      return "CAM_VGA";
    case str2int("1280x720"):
      return "CAM_HD";
    case str2int("1920x1080"):
      return "CAM_FHD";
    case str2int("2208x1242"):
      return "CAM_2K";
    default:
      return "NOT SUPPORTED RESOLUTION";
  }
}

/*
* Projects a 3D point in camera plane
*/
cv::Point2d project(const cv::Mat & intrinsics, const cv::Vec3d & p3d)
{   

  double z = p3d[2];
  //double z = 1.0;

  double fx = intrinsics.at<double>(0,0);
  double fy = intrinsics.at<double>(1,1);
  double cx = intrinsics.at<double>(0,2);
  double cy = intrinsics.at<double>(1,2);

  return cv::Point2d((p3d[0]*fx/z+cx), (p3d[1]*fy/z +cy));
}

/*
* Transofms a vector of 2D points into a cv::Mat with 2 channels and 1 row
*/
void vector2Mat(const std::vector<cv::Point2d> & points, cv::Mat & pmat)
{

  pmat = cv::Mat(1,points.size(),CV_64FC2);

  for (int i = 0; i < points.size(); i++)
  {
    pmat.at<cv::Point2d>(0,i) = points[i];
  }
}

void mat2Vector(const cv::Mat & mat, std::vector<cv::Point2f> & bodyparts)
{
  for(int i = 0; i < mat.cols; i++)
  {
    cv::Point3f pp = mat.at<cv::Point3d>(0,i);
    cv::Point2f p(pp.x,pp.y);
    bodyparts.push_back(p);
  }
}

/*
*TODO: check implementation on openpose library. There exists for sure.
*/
void opArray2Mat(const op::Array<float> & keypoints, cv::Mat & campnts)
{

  double x = 0.0;
  double y = 0.0;
  double conf = 0.0;

  //Ugliest AND SLOWEST
  std::vector<std::string> spoints = CSVTokenize(keypoints.toString());

  int people = keypoints.getVolume()/54;

  campnts = cv::Mat(1,people*18,CV_64FC3);

  for (int i = 0; i < 54 * people; i += 3)
  {
    x = atof(spoints[i].c_str());
    y = atof(spoints[i+1].c_str());
    conf = atof(spoints[i+2].c_str());
    cv::Vec3d elem(x,y,conf);
    campnts.at<cv::Vec3d>(0,i/3) = elem;
  }
}


/* 
 *  Format string in CSV format 
 */
std::vector<std::string> CSVTokenize(std::string kpl_str)
{
  kpl_str.erase(0, kpl_str.find("\n") + 1);
  std::replace(kpl_str.begin(), kpl_str.end(), '\n', ' ');

  std::vector<std::string> vec;
  std::istringstream iss(kpl_str);
  copy(std::istream_iterator<std::string>(iss),std::istream_iterator<std::string>(),back_inserter(vec));

  return vec;
}

void getConfidences(const op::Array<float> & poseKeypoints, std::vector<float> & confidences)
{ 
  cv::Mat kpoints;
  opArray2Mat(poseKeypoints, kpoints);

  for(int i = 0; i < kpoints.cols; i++)
  {
    cv::Vec3d p3d = kpoints.at<cv::Vec3d>(0,i);
    confidences.push_back(p3d[2]);
  }
}

/*
*Produces a CSV string representing 3D body points in a single frame
*/
void emitCSV3D(std::ofstream & outputfile, const op::Array<float> & poseKeypointsL, int cur_frame, const cv::Mat &points3D)
{
   std::string kp_str = poseKeypointsL.toString();
   std::vector<std::string> tokens = CSVTokenize(kp_str);
   int camera = 0;

   std::vector<float> confidences;
   //getConfidences(poseKeypointsL,confidences);

   //if no person detected, output 54 zeros
   if (tokens.size() == 0)
   {
     outputfile << camera << " " << cur_frame << " " << 0 << " ";
     for (int j = 0; j < 54; j++)
     {
       outputfile << 0.000 << " ";
     }

     outputfile << '\n';
   }
  
  else
  {  
    for (int i = 0; i < (points3D.cols/18) + 1; i++)
    {

      outputfile << camera << " " << cur_frame << " " << i << " ";
      int offset = 18 * i;

      for(int j = offset; j < offset + 18; j++)
      {
        cv::Vec3d point = points3D.at<cv::Vec3d>(j);
        outputfile << point[0] << " " << point[1] << " " << point[2] <<" ";
      }

      outputfile << "\n";
    }

  }
}

/*
* Produces a CSV string representing body positions found in a single frame by one camera   
*/
void emitCSV(std::ofstream & outputfile, const op::Array<float> & poseKeypoints, int camera, int cur_frame)
{  
   std::string kp_str = poseKeypoints.toString();
   std::vector<std::string> tokens = CSVTokenize(kp_str);

   //if no person detected, output 54 zeros
   if (tokens.size() == 0)
   {
     outputfile << camera << " " << cur_frame << " " << 0 << " ";
     for (int j = 0; j < 54; j++)
     {
       outputfile << 0.000 << " ";
     }

     outputfile << '\n';
   }

  for (int i = 0; i < poseKeypoints.getVolume(); i += 54)
   {
     outputfile << camera << " " << cur_frame << " " << i/54 << " ";
     for (int j = 0; j < 54; j++)
     {
       outputfile << tokens[i+j] << " ";
     }

     outputfile << '\n';
   }  
}
/*
* Takes the matrix of detected body points and set the zero points to NaN points
*/
void filterVisible(const cv::Mat & pntsL, cv::Mat & nzL)
{
  cv::Vec2d pl;
  cv::Vec2d zerov(0.0,0.0);

  std::vector<cv::Vec2d> pntsl;

  for (int i = 0; i < pntsL.cols; i++)
  {
    pl = pntsL.at<cv::Vec2d>(0,i);

    if (pl != zerov)
    {
      pntsl.push_back(pl);
    }
    else
    {
      double n = std::numeric_limits<double>::quiet_NaN();
      pntsl.push_back(cv::Vec2d(n,n));
    }

  }

  nzL = cv::Mat(1,pntsl.size(),CV_64FC2);

  for (int i = 0; i < pntsl.size(); i++)
  {
    nzL.at<cv::Vec2d>(0,i) = pntsl[i];
  }
}

/*
* Takes the matrix of points got from two different points of view. When there is at leas a zero point in a couple of 
* corresponding points, sets both of them to NaN 
*/
void filterVisible(const cv::Mat & pntsL, const cv::Mat & pntsR, cv::Mat & nzL, cv::Mat & nzR)
{ 
  cv::Vec2d pl;
  cv::Vec2d pr;
  cv::Vec2d zerov(0.0,0.0);

  std::vector<cv::Vec2d> pntsl;
  std::vector<cv::Vec2d> pntsr;

  for (int i = 0; i < pntsL.cols; i++)
  {
    pl = pntsL.at<cv::Vec2d>(0,i);
    pr = pntsR.at<cv::Vec2d>(0,i);


    if (pl != zerov && pr != zerov)
    {
      pntsl.push_back(pl);
      pntsr.push_back(pr);
    }
    else
    {
      //Push Nan instead of zero
      double n = std::numeric_limits<double>::quiet_NaN();
      pntsl.push_back(cv::Vec2d(n,n));
      pntsr.push_back(cv::Vec2d(n,n));
    }

  }

  nzL = cv::Mat(1,pntsl.size(),CV_64FC2);
  nzR = cv::Mat(1,pntsr.size(),CV_64FC2);

  for (int i = 0; i < pntsl.size(); i++)
  {
    nzL.at<cv::Vec2d>(0,i) = pntsl[i];
    nzR.at<cv::Vec2d>(0,i) = pntsr[i];
  }
}

void filterVisible(std::vector<cv::Mat> & bodies_left, std::vector<cv::Mat> & bodies_right, double conf)
{  
  cv::Vec3d pl;
  cv::Vec3d pr;
  cv::Vec2d zerov(0.0,0.0);
  double n = std::numeric_limits<double>::quiet_NaN();

  for(int i = 0; i < bodies_left.size(); i++)
  {
    for (int j = 0; j < 18; j++)
    {

      cv::Vec3d pl = bodies_left[i].at<cv::Vec3d>(0,j);
      cv::Vec3d pr = bodies_right[i].at<cv::Vec3d>(0,j);

      cv::Vec2d ppl = cv::Vec2d(pl[0], pl[1]);
      cv::Vec2d ppr = cv::Vec2d(pr[0], pr[1]);

      if(ppl == zerov || ppr == zerov)
      {
        //Push Nan instead of zero
        bodies_left[i].at<cv::Vec3d>(0,j) = cv::Vec3d(n,n,n);
        bodies_right[i].at<cv::Vec3d>(0,j) = cv::Vec3d(n,n,n);
      }

      //check also confidence
      if(pl[2] < conf || pr[2] < conf)
      {
        bodies_left[i].at<cv::Vec3d>(0,j) = cv::Vec3d(n,n,n);
        bodies_right[i].at<cv::Vec3d>(0,j) = cv::Vec3d(n,n,n);
      }

    }
  }
}

/*
* Transforms the type of a cv::Mat into a string
*/
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

/*
* Draws detected bodypoints as circle in the image
*/
void drawPoints(const cv::Mat & points, cv::Mat & image)
{
  for(int i = 0; i < points.cols; i++)
  { 
    cv::Point3d d = points.at<cv::Point3d>(0,i);
    cv::Point2d c = cv::Point2d(d.x,d.y);
    cv::circle(image,c,4,cv::Scalar(255,0,0),2);
  }
}

/*
* Takes a stereo image and splits it in two
*/
void splitVertically(const cv::Mat & input, cv::Mat & outputleft, cv::Mat & outputright)
{

  int rowoffset = input.rows;
  int coloffset = input.cols / 2;

  int r = 0;
  int c = 0;

  outputleft = input(cv::Range(r, std::min(r + rowoffset, input.rows)), cv::Range(c, std::min(c + coloffset, input.cols)));

  c += coloffset;

  outputright = input(cv::Range(r, std::min(r + rowoffset, input.rows)), cv::Range(c, std::min(c + coloffset, input.cols)));
}

/*
* Takes the cv::Mat generated by OpenPose and converts it in a vector of cv::Mat whose size is equal 
* to the number of detected bodies
*/
void pts2VecofBodies(const cv::Mat & pts1, std::vector<cv::Mat> & bodies_left)
{
  for(int i=0; i < pts1.cols/18; i++)
  {
    cv::Mat pleft(1,18,CV_64FC3);

    for (int j=0; j<18; j++)
    {
      pleft.at<cv::Vec3d>(0,j) = pts1.at<cv::Vec3d>(0,(18*i)+j);
    }

    bodies_left.push_back(pleft);
  }
}

void vecofBodies2Pts(const std::vector<cv::Mat> bodies, cv::Mat & pts)
{
  pts = cv::Mat(1,18*bodies.size(),CV_64FC2);

  for(int i = 0; i < bodies.size(); i++)
  {
    for(int j = 0; j < 18; j++)
    { 
      cv::Vec3d pt = bodies[i].at<cv::Vec3d>(0,j);
      pts.at<cv::Vec2d>(0, j + (18*i)) = cv::Vec2d(pt[0],pt[1]);
    }
  }
}

void body2VecofPoints(const cv::Mat & body, std::vector<cv::Point2d> & v)
{

  std::vector<cv::Point3d> v3d;
  for( int i = 0; i < body.cols; i++)
  {
    v3d.push_back(body.at<cv::Point3d>(0,i));
  }

  for (auto p : v3d)
  {
    v.push_back(cv::Point2d(p.x,p.y));
  }
}


double computeDiff(const cv::Mat & ml, const cv::Mat & mr, int offset, int cn = 4)
{
  //TODO: get only the bodypoints in common
  std::vector<cv::Point2d> pl;
  std::vector<cv::Point2d> pr;

  cv::Point2d offpoint((double)offset, 0.0);

  body2VecofPoints(ml,pl);
  body2VecofPoints(mr,pr);

  double acc = 0.0;
  int i = 0;
  int j = 0;

  while(i < cn && j < 18)
  {
    if (pl[j] != cv::Point2d(0,0) && pr[j] != cv::Point2d(0,0))
    {
      acc = acc + cv::norm((pl[j]+offpoint)-pr[j]);
      i++;
    }
    j++;
  }

  return acc; 
}

void associate(const std::vector<cv::Mat> & bodies_left, const std::vector<cv::Mat> & bodies_right, std::vector<int> & minindsL, int offset = 0)
{

  minindsL.resize(bodies_left.size());

  for(int i = 0; i < bodies_left.size(); i++)
  {

    int minind = -1;
    double mindiff = 9999999;

    for(int j = 0; j < bodies_right.size(); j++)
    { 
      double diff = computeDiff(bodies_left[i], bodies_right[j], offset);
      if( diff < mindiff)
      {
        mindiff = diff;
        minind = j; 
      }
    } 

    minindsL[i] = minind;
  }
}

/*
* Find Correspondent bodies from camera left and right. Note that pts1 and pts2 contain also the confidence level
*/
void findCorrespondences(const cv::Mat & pts1, const cv::Mat & pts2, cv::Mat & sorted_left, cv::Mat & sorted_right)
{

  //TODO: divide the points in bodies -> every 18 points one body, get the center of each body
  std::vector<cv::Mat> bodies_left;
  std::vector<cv::Mat> bodies_right;

  pts2VecofBodies(pts1, bodies_left);
  pts2VecofBodies(pts2, bodies_right);

  std::vector<int> minindsL;
  std::vector<int> minindsR;

  associate(bodies_left, bodies_right, minindsL);
  associate(bodies_right, bodies_left, minindsR);

  /*
  std::cout << "Min inds L " << std::endl;
  for (auto e : minindsL)
  {
    std::cout << e << " ";
  }
  std::cout << "\n";

  std::cout << "Min inds R " << std::endl;
  for (auto e : minindsR)
  {
    std::cout << e << " ";
  }
  std::cout << "\n";
  */


  //TODO: find conflicts: minindsL[i] contains the index of the body associated with i with respet to i
  std::map<int,int> corresp;
  for (int i = 0; i < minindsL.size(); i++)
  {
    if( minindsR[minindsL[i]] == i)
    {
      corresp.insert(std::pair<int,int>(minindsL[i], i));
    }
  }

  std::vector<cv::Mat> proper_left;
  std::vector<cv::Mat> proper_right;

  for(std::map<int,int>::iterator it = corresp.begin(); it != corresp.end(); ++it)
  {
   // std::cout << "associating left " << it->second << " with right " << it->first << std::endl;
    //std::cout << "left size: " << bodies_left.size() << " right size: " << bodies_right.size() << std::endl;
    proper_right.push_back(bodies_right[it->first]);
    proper_left.push_back(bodies_left[it->second]);
  }

  //If zeros in at least one: put NaN, NaN in both so that triangulation is meaningless 
  filterVisible(proper_left, proper_right);

  vecofBodies2Pts(proper_left, sorted_left);
  vecofBodies2Pts(proper_right, sorted_right);

}

/*
Gaussian kernel
1 4 7 4 1
4 16 26 16 4
7 26 41 26 7
4 16 26 16 4
1 4 7 4 1
*/

double MaxPool(const cv::Mat & matrix)
{ 

  double min,max = 0.0;  
  cv::minMaxLoc(matrix,&min,&max);

  return max;
}

double MinPool(const cv::Mat & matrix)
{

  double min,max = 0.0;
  cv::minMaxLoc(matrix,&min,&max);

  return min;
}

double AvgPool(const cv::Mat & matrix)
{

  double sum = cv::sum(matrix)[0];
  int nonzero = cv::countNonZero(matrix); 

  return sum / (double)nonzero;
}

double MinPoolDepth(cv::Mat & matrix)
{

  for(int i = 0; i < matrix.rows; i++)
  {
    for (int j= 0; j < matrix.cols; j++)
    {
      if(matrix.at<uint16_t>(i,j) == 0)
      {
        matrix.at<uint16_t>(i,j) = 65535;
      }
    }
  }

  /*std::cout << "kernel " << std::endl;
  std::cout << matrix << std::endl;*/

  double min = MinPool(matrix);

  if(min == 65535)
  {
    min = 0.0;
  }

  return min;
}

cv::Mat createGaussianFilter(int kernelsize, double sigma = 1.0)
{
    // set standard deviation to 1.0
    cv::Mat kernel = cv::Mat(kernelsize, kernelsize, CV_64FC1);
    double r, s = 2.0 * sigma * sigma;
 
    // sum is for normalization
    double sum = 0.0;

    int hsize = kernelsize/2;
 
    // generate kxk kernel
    for (int x = -hsize; x <= hsize; x++)
    {
        for(int y = -hsize; y <= hsize; y++)
        {
            r = sqrt(x*x + y*y);
            kernel.at<double>(x + hsize,y + hsize) = (exp(-(r*r)/s))/(M_PI * s);
            sum += kernel.at<double>(x + hsize,y + hsize);
        }
    }

    // normalize the Kernel
    for(int i = 0; i < kernelsize; ++i)
    {
        for(int j = 0; j < kernelsize; ++j)
        {
            kernel.at<double>(i,j) /= sum;
            kernel.at<double>(i,j) *= 270.0;
        }
    }

    return kernel;
}

double gaussianAvg(cv::Mat & matrix)
{
  //TODO: get gaussian kernel
  if(matrix.rows % 2 == 0)
  {
    return 0.0;
  }
  cv::Mat gKernel = createGaussianFilter(matrix.rows);
  std::cout << matrix << std::endl;

  double csum = cv::sum(gKernel)[0];
  double esum = 0.0;

  for(int x = 0; x < matrix.rows; x++)
  {
    for(int y = 0; y < matrix.cols; y++)
    {
      double tmp = (double)matrix.at<uint16_t>(x,y);
      esum = esum + (tmp * gKernel.at<double>(x,y));

      if(tmp == 0.0)
      {
        csum -= gKernel.at<double>(x,y);
      }

    }
  }
  std::cout << "csum " << csum << std::endl;
  std::cout << "esum " << esum << std::endl;
  std::cout << "esum/csum " << esum/csum << std::endl;
  return (esum/csum)/4;
}


double Pool(const cv::Mat & disp, int u, int v, int side, std::function<double(cv::Mat &)> function, cv::Mat & kernel)
{ 

  if(side % 2 == 0)
  {
    std::cout << "even kernel size not permitted " << std::endl;
    exit(-1);
  }

  double wlb,hlb;
  double wside,hside;

  wlb = std::max(0,u - side/2);
  hlb = std::max(0,v - side/2);

  wside = std::min(disp.rows - u, side);
  hside = std::min(disp.cols - v, side);

  int nside = std::min(wside,hside);

  /*
  std::cout << "creating matrix" << std::endl;
  std::cout << "rows: " << disp.rows << " cols: " << disp.cols << std::endl;
  std::cout << "u: " << u << " v: " << v << std::endl;
  std::cout << "wlb: " << wlb << " hlb: " << hlb << " wside: " << wside << " hside: " << hside << std::endl;
  */
  cv::Mat matrix(disp(cv::Rect(hlb,wlb,nside,nside)));
  
  kernel = matrix;

  return function(matrix);
}

void PoseProcess(const OpenPoseParams & params, const cv::Mat & inputImage, op::Array<float> & poseKeypoints, cv::Mat & outputImage)
{
                

  const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
   // Step 2 - Get desired scale sizes
   std::vector<double> scaleInputToNetInputs;
   std::vector<op::Point<int>> netInputSizes;
   double scaleInputToOutput;
   op::Point<int> outputResolution;
   std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
       = params.scaleAndSizeExtractor_->extract(imageSize);
   // Step 3 - Format input image to OpenPose input and output formats
   const auto netInputArray = params.cvMatToOpInput_.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
   auto outputArray = params.cvMatToOpOutput_.createArray(inputImage, scaleInputToOutput, outputResolution);
   // Step 4 - Estimate poseKeypoints
   params.poseExtractorCaffe_->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
   poseKeypoints = params.poseExtractorCaffe_->getPoseKeypoints();
   // Step 5 - Render poseKeypoints
   params.poseRenderer_->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
   // Step 6 - OpenPose output format to cv::Mat
   outputImage = params.opOutputToCvMatL_.formatToCvMat(outputArray);

}

bool trackLK(const cv::Mat & previous, const cv::Mat & current, std::vector<cv::Point2f> & ps, std::vector<cv::Point2f> & pe,
              double max_error, cv::Mat & output)
{
   std::vector<uchar> status;
   std::vector<float> err;

   //std::cout << "difference " << cv::norm(previous - current) << std::endl;

   //TODO: problem do not track points equal to 0
   cv::calcOpticalFlowPyrLK(previous, current, ps, pe, status, err);

   output = cv::Mat(1,ps.size(), CV_64FC3);

   for(int i = 0; i < output.cols; i++)
   {  
    if(ps[i].x != 0.0 || ps[i].y != 0.0)
    {
      output.at<cv::Point3d>(0,i) = cv::Point3d(pe[i].x, pe[i].y, 1.0);
    }
    else
    {
      output.at<cv::Point3d>(0,i) = cv::Point3d(0.0,0.0,0.0);
    }

    if(pe[i].x > current.cols || pe[i].y > current.rows)
    {
      return false;
    }
   }

  std::vector<cv::Point2f> tmp[2];
  tmp[0] = ps;
  tmp[1] = pe;

  //calculate MSE tracking backwards
  cv::calcOpticalFlowPyrLK(current,previous, tmp[1], tmp[0], status, err);

   //TODO: get the error between tmp[0] and points_[0]
   double cur_error = 0.0;
   for(int i = 0; i < tmp[0].size(); i++)
   {
    cur_error = cv::norm((tmp[0][i] - ps[i]));
    if(cur_error > max_error)
    {
      return false;
    }
   }

  for(int i = 0; i < ps.size(); i++)
  {
    if(ps[i].x != 0.0 && ps[i].y != 0.0)
    {
      ps[i] = pe[i];
    }
    else
    {
      ps[i] = cv::Point2f(0.0,0.0);
    }
  } 

  return true;
}

