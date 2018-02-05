#include "Picture_Tool.h"

void Picture_Tool::showNRGB(char *path)
{
	IplImage *img;
	img = cvLoadImage(path);
	cvNamedWindow("showNRGB", CV_WINDOW_AUTOSIZE);
	cvShowImage("showNRGB", NormalizeImage(img));
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("showNRGB");
}

void Picture_Tool::saveNRGB(char *spath, char *tpath) 
{
	IplImage *img;
	img = cvLoadImage(spath);
	cout << "保存归一化图片" << endl;
	cvSaveImage(tpath, NormalizeImage(img));
	cvReleaseImage(&img);
}
//彩色图像的直方图均衡化  
IplImage* Picture_Tool::EqualizeHistColorImage(IplImage *pImage)
{
	IplImage *pEquaImage = cvCreateImage(cvGetSize(pImage), pImage->depth, 3);

	// 原图像分成各通道后再均衡化,最后合并即彩色图像的直方图均衡化  
	const int MAX_CHANNEL = 4;
	IplImage *pImageChannel[MAX_CHANNEL] = { NULL };

	int i;
	for (i = 0; i < pImage->nChannels; i++)
		pImageChannel[i] = cvCreateImage(cvGetSize(pImage), pImage->depth, 1);

	cvSplit(pImage, pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3]);

	for (i = 0; i < pImage->nChannels; i++)
		cvEqualizeHist(pImageChannel[i], pImageChannel[i]);

	cvMerge(pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3], pEquaImage);

	for (i = 0; i < pImage->nChannels; i++)
		cvReleaseImage(&pImageChannel[i]);

	return pEquaImage;
}
IplImage* Picture_Tool::NormalizeImage(IplImage *img)
{
	//1创建归一化的图像
	IplImage* imgavg = cvCreateImage(cvGetSize(img), 8, 3);
	
	//2获取图像高度和宽度信息，设置epslon的目的是防止除0的操作产生；
	int width = img->width;
	int height = img->height;
	int redValue, greenValue, blueValue;
	double sum, epslon = 0.000001;

	//3计算归一化的结果，并替换掉原像素值；
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			CvScalar src = cvGet2D(img, y, x);
			redValue = src.val[2];
			greenValue = src.val[1];
			blueValue = src.val[0];
			// 加上epslon，为了防止除以0的情况发生
			sum = redValue + greenValue + blueValue + epslon;
			CvScalar des = cvScalar(blueValue / sum * 255, greenValue / sum * 255, redValue / sum * 255, src.val[4]);
			cvSet2D(imgavg, y, x, des);

		}
	}

	//4返回归一化后的图像；
	return imgavg;
}


IplImage* Picture_Tool::twoValueImage(IplImage *img,int threshold){
	//1创建归一化的图像
	IplImage* imgavg = cvCreateImage(cvGetSize(img), 8, 3);
	//cvCvtColor(img, imgavg, CV_RGB2GRAY);
	//cvShowImage("test", imgavg);
	//2获取图像高度和宽度信息，设置epslon的目的是防止除0的操作产生；
	int width = img->width;
	int height = img->height;
	int redValue, greenValue, blueValue;

	//3计算归一化的结果，并替换掉原像素值；
	double fr = 0.0, fb = 0.0;
	uchar* piexl = new uchar;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			CvScalar src = cvGet2D(img, y, x);
			redValue = src.val[2];
			greenValue = src.val[1];
			blueValue = src.val[0];
			fr = two(redValue, greenValue, blueValue);
			if (fr>0.12){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}
			//cout << redValue << " " << greenValue << " " << blueValue << endl;
			//归一化阈值分割红色
			/*if (redValue>102 && greenValue<76.5){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			//归一化阈值分割蓝色
			/*if (blueValue>102){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			//RGB差分阈值分割红色
			/*if (redValue - greenValue>20.4 && redValue - blueValue>20.4){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			//RGB差分阈值分割蓝色
			/*if (blueValue - greenValue>2.55 && blueValue-redValue >2.55){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			else{
					CvScalar des = cvScalar(0, 0, 0, src.val[4]);
					cvSet2D(imgavg, y, x, des);	
				}
			
		}
	}
	//cvSaveImage("gray.jpg", imgavg);
	////4返回归一化后的图像；
	return imgavg;
}
double Picture_Tool::two(double x, double y, double z){
	double sum = x + y + z;
	return __max(0, __min(x - y, x - z) / sum);
}

/*
将图片从RGB转到HSV颜色空间
*/
void Picture_Tool::RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
{

	double r, g, b;
	double h, s, i;

	double sum;
	double minRGB, maxRGB;
	double theta;

	r = red / 255.0;
	g = green / 255.0;
	b = blue / 255.0;

	minRGB = ((r<g) ? (r) : (g));
	minRGB = (minRGB<b) ? (minRGB) : (b);

	maxRGB = ((r>g) ? (r) : (g));
	maxRGB = (maxRGB>b) ? (maxRGB) : (b);

	sum = r + g + b;
	i = sum / 3.0;

	if (i < 0.001 || maxRGB - minRGB < 0.001)
	{
		h = 0.0;
		s = 0.0;
		//return ;  
	}
	else
	{
		s = 1.0 - 3.0*minRGB / sum;
		theta = sqrt((r - g)*(r - g) + (r - b)*(g - b));
		theta = acos((r - g + r - b)*0.5 / theta);
		if (b <= g)
			h = theta;
		else
			h = 2 * PI - theta;
		if (s <= 0.01)
			h = 0;
	}

	hue = (int)(h * 180 / PI);
	saturation = (int)(s * 100);
	intensity = (int)(i * 100);
}

void Picture_Tool::FillHole(const Mat srcBw, Mat &dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);
}

bool Picture_Tool::isCircle(const Mat srcBw, Mat& mytemp)//（待改进）
{//输入的是一个灰度图像
	Mat temp = Mat::zeros(srcBw.size(), CV_8UC1);
	bool iscircle = false;
	//获得srcBw信息
	int w = srcBw.cols;
	int h = srcBw.rows;
	int w1 = mytemp.cols;
	int h1 = mytemp.rows;
	//cout << w << " " << w1 << " " << h << " " << h1 << endl;
	int count1 = 0;//各部分的缺失像素计数器
	int count2 = 0;
	int count3 = 0;
	int count4 = 0;
	//将srcBw平均分成四份,进行访问缺失的像素个数、所占比重
	//先访问左上
	for (int i = 0; i < h / 2; i++)
	{
		for (int j = 0; j < w / 2; j++)
		{
			if (srcBw.at<uchar>(i, j) == 0)
			{
				temp.at<uchar>(i, j) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 0) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 1) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 2) = 255;
				count1++;
			}
		}
	}
	//右上
	for (int i = 0; i < h / 2; i++)
	{
		for (int j = w / 2 - 1; j < w; j++)
		{
			if (srcBw.at<uchar>(i, j) == 0)
			{
				temp.at<uchar>(i, j) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 0) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 1) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 2) = 255;
				count2++;
			}
		}
	}
	//左下
	for (int i = h / 2 - 1; i < h; i++)
	{
		for (int j = 0; j < w / 2; j++)
		{
			if (srcBw.at<uchar>(i, j) == 0)
			{
				temp.at<uchar>(i, j) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 0) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 1) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 2) = 255;
				count3++;
			}
		}
	}
	//右下
	for (int i = h / 2 - 1; i < h; i++)
	{
		for (int j = w / 2 - 1; j < w; j++)
		{
			if (srcBw.at<uchar>(i, j) == 0)
			{
				temp.at<uchar>(i, j) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 0) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 1) = 255;
				mytemp.at<uchar>(i, j*mytemp.channels() + 2) = 255;
				count4++;
			}
		}
	}


	float c1 = (float)count1 / (float)(w*h);//左上
	float c2 = (float)count2 / (float)(w*h);//右上
	float c3 = (float)count3 / (float)(w*h);//左下
	float c4 = (float)count4 / (float)(w*h);//右下
	//imshow("temp",mytemp);
	cout << "result: " << c1 << "," << c2
		<< "," << c3 << "," << c4 << endl;

	//限定每个比率的范围
	if ((c1>0.037&&c1<0.12) && (c2>0.037&&c2<0.12) && (c2>0.037&&c2<0.12) && (c2>0.037&&c2<0.12))
	{
		//限制差值,差值比较容错，相邻块之间差值相近，如左上=右上&&左下=右下或左上=左下&&右上=右下
		if ((abs(c1 - c2)<0.04&&abs(c3 - c4)<0.04) || (abs(c1 - c3)<0.04&&abs(c2 - c4)<0.04))
		{
			iscircle = true;
		}
	}


	return iscircle;
}