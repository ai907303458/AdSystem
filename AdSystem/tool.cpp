#include "tool.h"

CvSVM classifier;//���������  
/*
	��ͼƬ��RGBת��HSV��ɫ�ռ�
*/
void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
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

/*
	û����
*/
void FillHole(const Mat srcBw, Mat &dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);
}

/*
	������ͼƬ�Ĵ�С��һ��
*/
void ReSize(int width, int height,int nImages){
	//����ͼ������������  
	char **img_name_index = new char *[nImages]; // load image names of Ukbench  
	ifstream inf_img("..\\data\\img_name.txt", ios_base::in);
	if (!inf_img) return;
	for (int n = 0; n < nImages; n++)
	{
		img_name_index[n] = new char[100];
		if (!img_name_index[n]) break;
		inf_img.getline(img_name_index[n], 32);
	}
	inf_img.close();
	for (int i = 0; i<nImages; i++)
	{
		cout << i + 1 << endl;
		char temp[100] = "\0";
		char result_img[100] = "\0";
		sprintf(temp, "..\\images\\%s", img_name_index[i]);
		IplImage *src;
		src = cvLoadImage(temp);

		IplImage* gray_image;
		gray_image = cvCreateImage(cvGetSize(src), 8, 1);
		cvCvtColor(src, gray_image, CV_BGR2GRAY);

		IplImage* norm_image;
		CvSize norm_cvsize;
		norm_cvsize.width = width;  //Ŀ��ͼ��Ŀ�      
		norm_cvsize.height = height; //Ŀ��ͼ��ĸ�    

		norm_image = cvCreateImage(norm_cvsize, gray_image->depth, gray_image->nChannels); //����Ŀ��ͼ��    
		cvResize(gray_image, norm_image, CV_INTER_LINEAR); //����Դͼ��Ŀ��ͼ��   
		sprintf(result_img, "..\\new_image\\%d.jpg", i + 1);
		cvSaveImage(result_img, norm_image);//cvSaveImage("norm.jpg", norm_image); //�����һ��ͼ�񵽱����ļ�����  
		cvReleaseImage(&norm_image);
		cvReleaseImage(&src);
		cvReleaseImage(&gray_image);
	}
}

/*
	��һ��RGB�ļ������ء�
*/
IplImage* NormalizeImage(IplImage *img)
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
			redValue = src.val[0];
			greenValue = src.val[1];
			blueValue = src.val[2];
			// ����epslon��Ϊ�˷�ֹ����0���������
			sum = redValue + greenValue + blueValue + epslon;
			CvScalar des = cvScalar(redValue / sum * 255, greenValue / sum * 255, blueValue / sum * 255, src.val[4]);
			cvSet2D(imgavg, y, x, des);

		}
	}

	//4���ع�һ�����ͼ��
	return imgavg;
}

/*
	���������
*/
void LoadSVM(char *train_path){

	cout << "����SVMѵ�����" << endl;
	classifier.load(train_path);//·�� 
	//classifier.load("F:/picture/GTSRB/Final_Training/train.xml");//·�� 
	cout << "������ɣ���ʼ����ͼƬ" << endl;
}
/*
	��һ��ͼƬ���з���
*/
void RGB2HSV_SVM(char *path){
	
	Mat src = imread(path);
	Mat copy;
	src.copyTo(copy);
	int width = src.cols;   //ͼ����  
	int height = src.rows;   //ͼ��߶�  
	//ɫ�ʷָ�  
	double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
	Mat matRgb = Mat::zeros(src.size(), CV_8UC1);
	Mat Mat_rgb_copy;//һ���ݴ浥Ԫ  
	int x, y;
	for (y = 0; y < height; y++)
	{
		for (x = 0; x<width; x++)
		{
			B = src.at<Vec3b>(y, x)[0];
			G = src.at<Vec3b>(y, x)[1];
			R = src.at<Vec3b>(y, x)[2];
			RGB2HSV(R, G, B, H, S, V);
			//��ɫ��337-360  
			if ((H >= 337 && H <= 360 || H >= 0 && H <= 10) && S >= 12 && S <= 100 && V>20 && V < 99)
			{
				matRgb.at<uchar>(y, x) = 255;
			}
		}
	}
	//imshow("hsi",Mat_rgb);  
	//imshow("Mat_rgb",Mat_rgb);  
	medianBlur(matRgb, matRgb, 3);
	//imshow("medianBlur", Mat_rgb);  
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
	erode(matRgb, matRgb, element);//��ʴ  
	//imshow("erode", Mat_rgb);  
	dilate(matRgb, matRgb, element1);//����  
	//imshow("dilate", Mat_rgb);  
	FillHole(matRgb, matRgb);//���   
	//imshow("fillHole", Mat_rgb);  
	matRgb.copyTo(Mat_rgb_copy);
	vector<vector<Point> > contours;//����  
	vector<Vec4i> hierarchy;//�ֲ�  
	findContours(matRgb, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// ����αƽ����� + ��ȡ���κ�Բ�α߽��  
	vector<vector<Point> > contours_poly(contours.size());//���ƺ�������㼯   
	vector<Rect> boundRect(contours.size()); //��Χ�㼯����С����vector    
	vector<Point2f>center(contours.size());//��Χ�㼯����СԲ��vector   
	vector<float>radius(contours.size());//��Χ�㼯����СԲ�ΰ뾶vector   

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//�Զ�����������ʵ����ƣ�contours_poly[i]������Ľ��Ƶ㼯  
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//���㲢���ذ�Χ�����㼯����С����   
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);//���㲢���ذ�Χ�����㼯����СԲ�μ���뾶  
	}
	Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
	int count = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundRect[i];
		//cout << rect<<endl;  
		//�߿������  
		float ratio = (float)rect.width / (float)rect.height;
		//�������       
		float Area = (float)rect.width * (float)rect.height;
		float dConArea = (float)contourArea(contours[i]);
		float dConLen = (float)arcLength(contours[i], 1);
		if (dConArea <400)
			continue;
		if (ratio>2 || ratio < 0.5)
			continue;

		//����Բɸѡ��ͨ���Ŀ��ȱʧ���رȽ�  
		Mat roiImage;
		Mat_rgb_copy(rect).copyTo(roiImage);
		//imshow("roiImage",roiImage);  
		//imshow("test",roiImage);  
		Mat temp;
		copy(rect).copyTo(temp);
		//imshow("test2",temp);//��ʾ�ӳ���ͼ����ȡ���ı�ʶ�����š�  

		copy(rect).copyTo(roiImage);
		//*********svm*********  
		Mat temp2 = Mat::zeros(temp.size(), CV_8UC1);
		cvtColor(temp, temp2, CV_BGR2GRAY);
		//resize(temp2, temp2, Size(48, 48));  
		resize(temp2, temp2, Size(30, 30));//30*30=900  
		temp2 = temp2.reshape(0, 1);
		temp2.convertTo(temp2, CV_32F);
		cout << temp2.size() << endl;

		int result = (int)classifier.predict(temp2) - 1;//svmԤ��  
		Scalar color = (0, 0, 255);//��ɫ�߻�����   
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//putText(src, labelname[result], cvPoint(boundRect[i].x, boundRect[i].y - 10), 1, 1, CV_RGB(255, 0, 0), 2);//��ɫ����ע��  
		//circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );  
		//sprintf_s(path, "E:\\vs2013\\opencv_code\\GTSRBtrafficSign\\extractAndPredict\\image\\result/%d_%d.jpg", k, count1);  
		//sprintf_s(path, "F:\\picture\\GTSRB\\Final_Test\\Test_Result\\%d_%d.jpg", 1, count++);
		imwrite(path, src);//�������յļ��ʶ����
	}
	cout << "����ͼƬ���" << endl; 
}
