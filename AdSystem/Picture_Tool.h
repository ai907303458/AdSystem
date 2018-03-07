#pragma once
#include<opencv2/opencv.hpp>  
#include <iostream>      
#include <fstream>  
#include <vector>    
#include <string>    
#include<time.h>  
#include<algorithm>
using namespace std;
using namespace cv;

#define PI 3.1415926  

class Picture_Tool 
{
public:
	void showNRGB(char *path);
	void saveNRGB(char *spath, char *tpath);
	IplImage* EqualizeHistColorImage(IplImage *pImage);
	IplImage* NormalizeImage(IplImage *img);
	IplImage* twoValueImage(IplImage *img, int type);
	double two(double R, double G, double B);
	IplImage* grayImpImage(IplImage *img);
	double adance(double x, double y, double z);
	void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity);
	void FillHole(const Mat srcBw, Mat &dstBw);
	bool Picture_Tool::isCircle(const Mat srcBw, Mat& mytemp);
};