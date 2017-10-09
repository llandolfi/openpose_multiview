#include "utilities.hpp"


int getHeight(const std::string & resolution)
{
  std::string height = resolution.substr(resolution.find("x") + 1);
  return atoi(height.c_str());
}

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

void vector2Mat(const std::vector<cv::Point2d> & points, cv::Mat & pmat)
{

  pmat = cv::Mat(1,points.size(),CV_64FC2);

  for (int i = 0; i < points.size(); i++)
  {
    pmat.at<cv::Point2d>(0,i) = points[i];
  }
}

/*
*TODO: check implementation on openpose library. There exists for sure.
*/
void opArray2Mat(const op::Array<float> & keypoints, cv::Mat & campnts)
{

  double x = 0.0;
  double y = 0.0;

  //Ugliest AND SLOWEST
  std::vector<std::string> spoints = CSVTokenize(keypoints.toString());

  int people = keypoints.getVolume()/54;

  campnts = cv::Mat(1,people*18,CV_64FC2);

  for (int i = 0; i < 54 * people; i += 3)
  {
    x = atof(spoints[i].c_str());
    y = atof(spoints[i+1].c_str());
    cv::Vec2d elem(x,y);
    campnts.at<cv::Vec2d>(0,i/3) = elem;
  }
}


/* Format string in CSV format 
 *
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

void emitCSV(std::ofstream & outputfile, std::string & kp_str, const op::Array<float> & poseKeypoints, int camera, int cur_frame)
{ 
   std::vector<std::string> tokens = CSVTokenize(kp_str);

   std::cout << "number of strings " << tokens.size() << std::endl;

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

  }

  nzL = cv::Mat(1,pntsl.size(),CV_64FC2);

  for (int i = 0; i < pntsl.size(); i++)
  {
    nzL.at<cv::Vec2d>(0,i) = pntsl[i];
  }
}

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

  }

  nzL = cv::Mat(1,pntsl.size(),CV_64FC2);
  nzR = cv::Mat(1,pntsr.size(),CV_64FC2);

  for (int i = 0; i < pntsl.size(); i++)
  {
    nzL.at<cv::Vec2d>(0,i) = pntsl[i];
    nzR.at<cv::Vec2d>(0,i) = pntsr[i];
  }
}


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

void drawPoints(const cv::Mat & points, cv::Mat & image)
{
  for(int i = 0; i < points.cols; i++)
  {
    cv::Point2d c = points.at<cv::Point2d>(0,i);
    cv::circle(image,c,4,cv::Scalar(255,255,255),2);
  }
}

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

void pts2VecofBodies(const cv::Mat & pts1, std::vector<cv::Mat> & bodies_left)
{
  for(int i=0; i < pts1.cols/18; i++)
  {
    cv::Mat pleft(1,18,CV_64FC2);

    for (int j=0; j<18; j++)
    {
      pleft.at<cv::Vec2d>(0,j) = pts1.at<cv::Vec2d>(0,(18*i)+j);
    }

    bodies_left.push_back(pleft);
  }
}

cv::Vec2d getMedian(const cv::Mat & body)
{

  double x,y = 0.0;
  double count = 0.0;

  for(int i = 0; i < body.cols; i++)
  {
    cv::Vec2d v = body.at<cv::Vec2d>(0,i);
    x = x + v[0];
    y = y + v[1];

    if (v[0] != 0.0 || v[1] != 0.0)
    {
      count++;
    }

  } 

  x = x / count;
  y = y / count;

  return cv::Vec2d(x,y);
}

int closestCentroidC(const cv::Vec2d & c, const std::vector<cv::Vec2d> & v)
{

  int minind = 0;
  double mindist = 999999999.9;
  int ind = -1;

  for (auto & a : v)
  {
    ind ++;
    double dist = std::sqrt(std::pow(c[0] - a[0], 2) + std::pow(c[1] - a[1], 2));

    if(dist < mindist)
    {
      mindist = dist;
      minind = ind;
    }
  }

  return minind;
}


int closestCentroidM(const cv::Vec2d & c, const std::vector<cv::Mat> & v)
{

  std::vector<cv::Vec2d> vc;

  for (auto & a : v)
  {
    vc.push_back(getMedian(a));
  }

  return closestCentroidC(c,vc);
}

void equalize(const cv::Mat & pts1, const cv::Mat & pts2, cv::Mat & outl, cv::Mat & outr)
{

  //TODO: divide the points in bodies -> every 18 points one body, get the center of each body
  std::vector<cv::Mat> bodies_left;
  std::vector<cv::Mat> bodies_right;

  std::vector<cv::Vec2d> centroids_left;
  std::vector<cv::Vec2d> centroids_right;

  std::vector<int> mininds;

  bool minatleft = true;

  pts2VecofBodies(pts1, bodies_left);
  pts2VecofBodies(pts2, bodies_right);

  for (auto & b : bodies_left)
  {
    centroids_left.push_back(getMedian(b));
  }

  for (auto & b : bodies_right)
  {
    centroids_right.push_back(getMedian(b));
  }

  std::vector<cv::Mat> * minbodies;
  std::vector<cv::Mat> * maxbodies;

  if(bodies_left.size() < bodies_right.size())
  {
    minbodies = &(bodies_left);
    maxbodies = &(bodies_right);
  }
  else
  {
    minatleft = false;
    minbodies = &(bodies_right);
    maxbodies = &(bodies_left);
  }

  for(auto & c : *minbodies)
  {
    //TODO:find the index of the closest element to c in maxbodies 
    std::vector<cv::Mat> topass = *maxbodies;
    std::vector<cv::Vec2d> vc;
    cv::Vec2d d = getMedian(c);

    for (auto & a : topass)
    {
      vc.push_back(getMedian(a));
    }

    int curmin = closestCentroidC(d,vc);
    mininds.push_back(curmin);
  }

  int outsize = minbodies->size();

  cv::Mat out1 = cv::Mat(1,outsize*18, CV_64FC2);
  cv::Mat out2 = cv::Mat(1,outsize*18, CV_64FC2);

  for (int i = 0; i < outsize; i++)
  {

    for(int j = 0; j < 18; j++)
    {
      out1.at<cv::Vec2d>(0,(18*i)+j) = minbodies->at(i).at<cv::Vec2d>(0,j);
      out2.at<cv::Vec2d>(0,(18*i)+j) = maxbodies->at(mininds[i]).at<cv::Vec2d>(0,j);
    }

  }

  if(minatleft)
  {
    outl = out1;
    outr = out2;
  }
  else
  {
    outl = out2;
    outr = out1;
  }
}

/*
* Returns the average of the rect of disp having side equal to side
*/
double smoothRect(const cv::Mat & disp, int u, int v, int side)
{

  std::cout << "parameters " << std::endl;

  std::cout << u << " " << v << " " << side << std::endl;

  double wlb,hlb;
  double wside,hside;

  wlb = std::max(0,u - side);
  hlb = std::max(0,v - side);

  wside = std::min(disp.cols - u, side);
  hside = std::min(disp.rows - v, side);

  std::cout << wlb << " " << hlb << " " << wside << " " << hside << std::endl;

  cv::Mat matrix(disp(cv::Rect(wlb,hlb,wside,hside)));

  double sum = cv::sum(matrix)[0];
  int nonzero = cv::countNonZero(matrix); 

  return sum / (double)nonzero;
}
