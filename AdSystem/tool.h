#include<opencv2/opencv.hpp>  
#include <iostream>      
#include <fstream>  
#include <vector>    
#include <string>    
#include<time.h>  

#define PI 3.1415926  

using namespace cv;
using namespace std;


void FillHole(const Mat srcBw, Mat &dstBw);
void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity);
void ReSize(int width, int height, int nImages);
IplImage* NormalizeImage(IplImage *img);
void LoadSVM(char *train_path);
void RGB2HSV_SVM(char *path);