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
	cout << "�����һ��ͼƬ" << endl;
	cvSaveImage(tpath, NormalizeImage(img));
	cvReleaseImage(&img);
}
//��ɫͼ���ֱ��ͼ���⻯  
IplImage* Picture_Tool::EqualizeHistColorImage(IplImage *pImage)
{
	IplImage *pEquaImage = cvCreateImage(cvGetSize(pImage), pImage->depth, 3);

	// ԭͼ��ֳɸ�ͨ�����پ��⻯,���ϲ�����ɫͼ���ֱ��ͼ���⻯  
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
	//1������һ����ͼ��
	IplImage* imgavg = cvCreateImage(cvGetSize(img), 8, 3);
	
	//2��ȡͼ��߶ȺͿ����Ϣ������epslon��Ŀ���Ƿ�ֹ��0�Ĳ���������
	int width = img->width;
	int height = img->height;
	int redValue, greenValue, blueValue;
	double sum, epslon = 0.000001;

	//3�����һ���Ľ�������滻��ԭ����ֵ��
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			CvScalar src = cvGet2D(img, y, x);
			redValue = src.val[2];
			greenValue = src.val[1];
			blueValue = src.val[0];
			// ����epslon��Ϊ�˷�ֹ����0���������
			sum = redValue + greenValue + blueValue + epslon;
			CvScalar des = cvScalar(blueValue / sum * 255, greenValue / sum * 255, redValue / sum * 255, src.val[4]);
			cvSet2D(imgavg, y, x, des);

		}
	}

	//4���ع�һ�����ͼ��
	return imgavg;
}


IplImage* Picture_Tool::twoValueImage(IplImage *img,int threshold){
	//1������һ����ͼ��
	IplImage* imgavg = cvCreateImage(cvGetSize(img), 8, 3);
	//cvCvtColor(img, imgavg, CV_RGB2GRAY);
	//cvShowImage("test", imgavg);
	//2��ȡͼ��߶ȺͿ����Ϣ������epslon��Ŀ���Ƿ�ֹ��0�Ĳ���������
	int width = img->width;
	int height = img->height;
	int redValue, greenValue, blueValue;

	//3�����һ���Ľ�������滻��ԭ����ֵ��
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
			//��һ����ֵ�ָ��ɫ
			/*if (redValue>102 && greenValue<76.5){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			//��һ����ֵ�ָ���ɫ
			/*if (blueValue>102){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			//RGB�����ֵ�ָ��ɫ
			/*if (redValue - greenValue>20.4 && redValue - blueValue>20.4){
				CvScalar des = cvScalar(255, 255, 255, src.val[4]);
				cvSet2D(imgavg, y, x, des);
			}*/
			//RGB�����ֵ�ָ���ɫ
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
	////4���ع�һ�����ͼ��
	return imgavg;
}
double Picture_Tool::two(double x, double y, double z){
	double sum = x + y + z;
	return __max(0, __min(x - y, x - z) / sum);
}

/*
��ͼƬ��RGBת��HSV��ɫ�ռ�
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

bool Picture_Tool::isCircle(const Mat srcBw, Mat& mytemp)//�����Ľ���
{//�������һ���Ҷ�ͼ��
	Mat temp = Mat::zeros(srcBw.size(), CV_8UC1);
	bool iscircle = false;
	//���srcBw��Ϣ
	int w = srcBw.cols;
	int h = srcBw.rows;
	int w1 = mytemp.cols;
	int h1 = mytemp.rows;
	//cout << w << " " << w1 << " " << h << " " << h1 << endl;
	int count1 = 0;//�����ֵ�ȱʧ���ؼ�����
	int count2 = 0;
	int count3 = 0;
	int count4 = 0;
	//��srcBwƽ���ֳ��ķ�,���з���ȱʧ�����ظ�������ռ����
	//�ȷ�������
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
	//����
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
	//����
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
	//����
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


	float c1 = (float)count1 / (float)(w*h);//����
	float c2 = (float)count2 / (float)(w*h);//����
	float c3 = (float)count3 / (float)(w*h);//����
	float c4 = (float)count4 / (float)(w*h);//����
	//imshow("temp",mytemp);
	cout << "result: " << c1 << "," << c2
		<< "," << c3 << "," << c4 << endl;

	//�޶�ÿ�����ʵķ�Χ
	if ((c1>0.037&&c1<0.12) && (c2>0.037&&c2<0.12) && (c2>0.037&&c2<0.12) && (c2>0.037&&c2<0.12))
	{
		//���Ʋ�ֵ,��ֵ�Ƚ��ݴ����ڿ�֮���ֵ�����������=����&&����=���»�����=����&&����=����
		if ((abs(c1 - c2)<0.04&&abs(c3 - c4)<0.04) || (abs(c1 - c3)<0.04&&abs(c2 - c4)<0.04))
		{
			iscircle = true;
		}
	}


	return iscircle;
}