#include "Picture_Tool.h"
#include "Video_Tool.h"

#define VIDEO_PATH "F:/picture/AVI/WKA00925_1.mp4"
#define SVM_PATH "F:/picture/Round_Sign/train.xml"
#define Train_PATH "F:/picture/Round_Sign/train.txt"
string labelname[500] = { "Around", "Forb_Drin", "Forb_Left", "Forb_Park", "Forb_right", "FororLeft", "Fororright", "Forward", "Left", "Right", "Splim100", "Splim120", "Splim20", "Splim30", "Splim50", "Splim60", "Splim70", "Splim80", "Turn_Left", "Turn_Right" };
RNG rng(12345);
CvSVM classifier;//����������



/*
ѵ��SVM��������
*/
void Train_SVM(char *train_path, char *result_path)
{
	int imgWidht = 48;//���¶���ͼƬ��С48*48  
	int imgHeight = 48;

	vector<string> imgTrainPath;//�����ļ�������     
	vector<int> imgTrainLabel;
	int nLine = 0;
	string buf;
	ifstream imagePath(train_path);//ѵ������λ��  
	unsigned long n;

	while (imagePath)//��ȡѵ��������imageName.txtһ��Ϊ·����һ��Ϊ��ǩ��ѭ��      
	{
		if (getline(imagePath, buf))
		{
			nLine++;
			if (nLine % 2 == 0)
			{
				imgTrainLabel.push_back(atoi(buf.c_str()));//atoi���ַ���ת�������ͣ�ͼƬ��ǩ      
			}
			else
			{
				imgTrainPath.push_back(buf);//ͼ��·��      
			}
		}
	}
	imagePath.close();

	CvMat *imgDataMat, *imgLabelMat;
	int nImgNum = nLine / 2; //��Ϊ��ÿ��һ�в���ͼƬ·��������Ҫ����2         
	imgDataMat = cvCreateMat(nImgNum, 900, CV_32FC1);  //cmd���������г���HOG dims ����descriptors.size()����С����ͬͼƬ��ֵ��һ�����������޸�    
	cvSetZero(imgDataMat);
	imgLabelMat = cvCreateMat(nImgNum, 1, CV_32FC1);
	cvSetZero(imgLabelMat);

	IplImage* srcImg;
	IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
	cout << "HOG������ʼ��ȡ" << endl;
	for (string::size_type i = 0; i != imgTrainPath.size(); i++)
	{
		srcImg = cvLoadImage(imgTrainPath[i].c_str(), 1);
		if (srcImg == NULL)
		{
			cout << " ͼƬ��ȡ���� " << imgTrainPath[i].c_str() << endl;
			continue;
		}
		cvResize(srcImg, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float>descriptors;//����Ľ��         
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //��ʼ����         
		cout << "HOG dims��descriptors.size()��: " << descriptors.size() << endl;
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(imgDataMat, i, n, *iter);//HOG�洢      
			n++;
		}
		cvmSet(imgLabelMat, i, 0, imgTrainLabel[i]);
	}
	cout << "HOG����������ȡ" << endl;

	CvSVM svm;//�½�SVM   
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);//����  
	cout << "svm��ʼѵ��" << endl;
	clock_t start = clock();
	svm.train(imgDataMat, imgLabelMat, NULL, NULL, param);//ѵ��svm      
	clock_t finish = clock();
	double consumeTime = (double)(finish - start);
	cout << "svmѵ������,��ʱ" << consumeTime << endl;
	svm.save(result_path);//ѵ������洢λ��  
	cvReleaseMat(&imgDataMat);
	cvReleaseMat(&imgLabelMat);

	system("pause");
	//return 0;
}

/*
ʹ��HOG������SVM���з��࣬����ͼƬ���Ѷ�λ�õı�־�ƣ�
�˺����������Ǽ������������Ч����
1������ͼƬ��·��������ͼƬ,
2������������浽predictResult.txt�ļ���
*/
void HOG_SVM_Recognize1()
{
	int imgWidht = 48;//���¶���ͼƬ��С48*48  
	int imgHeight = 48;
	IplImage *testImg;
	vector<string> testImgPath;//����ͼƬ��·��  
	vector<int> realTestImgLabel;
	vector<int> predictTestImgLabel;
	int predictRightNum = 0;
	double predictRightRatio;
	ifstream readTestImgPath("F:/picture/GTSRB/Final_Test/Test.txt");//��ȡ����ͼƬ·����txt�ļ�ΪͼƬ·������  
	string buf;
	int nLine = 0;
	cout << "ͼƬ·������" << endl;
	while (readTestImgPath)
	{
		if (getline(readTestImgPath, buf))
		{
			testImgPath.push_back(buf);//��·��  
		}
	}
	readTestImgPath.close();
	cout << "ͼƬ·���������" << endl;
	char line[512];
	ofstream predictResultPath("F:/picture/GTSRB/Final_Test/predictResult.txt");//Ԥ�����洢�ڴ��ı�     
	for (string::size_type j = 0; j != testImgPath.size(); j++)//��ȡ����ͼƬ      
	{
		cout << "��ȡ��" << j + 1 << "��ͼƬ" << endl;
		testImg = cvLoadImage(testImgPath[j].c_str(), 1);
		if (testImg == NULL)
		{
			cout << "ͼƬ��ȡ���� " << testImgPath[j].c_str() << endl;
			continue;
		}

		IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
		cvZero(trainImg);
		cvResize(testImg, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);//hog����ѵ��  
		vector<float> descriptors;//����Ľ��         
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //��ʼ����         
		cout << "HOG dims: " << descriptors.size() << endl;
		CvMat* svmTrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
		unsigned long n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(svmTrainMat, 0, n, *iter);
			n++;
		}
		CvSVM svm;//�½�SVM  
		svm.load("F:/picture/GTSRB/Final_Training/train.xml");
		int ret = svm.predict(svmTrainMat);
		predictTestImgLabel.push_back(ret);
		sprintf(line, "%s %d\r\n", testImgPath[j].c_str(), ret);
		predictResultPath << line;
		cout << "��" << j + 1 << "��ͼƬ�������" << endl;
	}
	predictResultPath.close();
	cout << "ͼƬ�������" << endl;
	system("pause");
}

/**
ʹ��HOG������SVM���з��ಢ����׼ȷ�ʣ��˺����������Ǽ������������Ч����
1������ͼƬ��·���ͱ�ǩ������ͼƬ,
2������������浽predictResult.txt�ļ��У�
3�����������ɹ���
*/
void HOG_SVM_Recognize2()
{
	int imgWidht = 48;//���¶���ͼƬ��С48*48  
	int imgHeight = 48;
	IplImage *testImg;
	vector<string> testImgPath;//����ͼƬ��·��  
	vector<int> realTestImgLabel;
	vector<int> predictTestImgLabel;
	int predictRightNum = 0;
	double predictRightRatio;
	ifstream readTestImgPath("E:\\vs2013\\opencv_code\\GTSRBtrafficSign\\test\\imageNameRandom.txt");//��ȡ����ͼƬ·����txt�ļ�ΪͼƬ·������  
	string buf;
	int nLine = 0;
	while (readTestImgPath)
	{
		if (getline(readTestImgPath, buf))
		{
			nLine++;
			if (nLine % 2 == 0)
				realTestImgLabel.push_back(atoi(buf.c_str()));//���ǩ  
			else
				testImgPath.push_back(buf);//��·��  
		}
	}
	readTestImgPath.close();

	char line[512];
	ofstream predictResultPath("E:\\vs2013\\opencv_code\\GTSRBtrafficSign\\test\\predictResultRandom.txt");//Ԥ�����洢�ڴ��ı�     
	for (string::size_type j = 0; j != testImgPath.size(); j++)//��ȡ����ͼƬ      
	{
		testImg = cvLoadImage(testImgPath[j].c_str(), 1);
		if (testImg == NULL)
		{
			cout << "ͼƬ��ȡ���� " << testImgPath[j].c_str() << endl;
			continue;
		}

		IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
		cvZero(trainImg);
		cvResize(testImg, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);//hog����ѵ��  
		vector<float> descriptors;//����Ľ��         
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //��ʼ����         
		cout << "HOG dims: " << descriptors.size() << endl;
		CvMat* svmTrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
		unsigned long n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(svmTrainMat, 0, n, *iter);
			n++;
		}

		CvSVM svm;//�½�SVM  
		svm.load("E:\\vs2013\\opencv_code\\GTSRBtrafficSign\\train\\train.xml");
		int ret = svm.predict(svmTrainMat);
		predictTestImgLabel.push_back(ret);
		sprintf(line, "%s %d\r\n", testImgPath[j].c_str(), ret);
		predictResultPath << line;
	}
	predictResultPath.close();

	for (string::size_type i = 0; i < realTestImgLabel.size(); i++)
	{
		if (realTestImgLabel[i] == predictTestImgLabel[i])//�ж�ʵ��ֵ��Ԥ��ֵ�Ƿ����  
			predictRightNum++;
	}
	predictRightRatio = (double)predictRightNum / predictTestImgLabel.size();//����Ԥ����ȷ�ı���  
	cout << "һ��Ԥ��" << predictTestImgLabel.size() << "��ͼƬ," << "��ȷ��Ϊ��" << predictRightRatio << endl;

	system("pause");
}

/*
	��һ��ͼƬ����ʶ��
*/
void HOG_SVM_Recog(char *img_path)
{
	int imgWidht = 48;//���¶���ͼƬ��С48*48  
	int imgHeight = 48;
	IplImage *testImg;
	testImg = cvLoadImage(img_path, 1);
	if (testImg == NULL)
	{
		cout << "ͼƬ��ȡ���� " << img_path << endl;
		exit(0);
	}

	IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
	cvZero(trainImg);
	cvResize(testImg, trainImg);
	//��ȡHOG����
	HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);//hog����ѵ��  
	vector<float> descriptors;//����Ľ��         
	hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //��ʼ����         
	cout << "HOG dims: " << descriptors.size() << endl;
	CvMat* svmTrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
	unsigned long n = 0;
	for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		cvmSet(svmTrainMat, 0, n, *iter);
		n++;
	}
	//��ͼƬ����Ԥ��
	int ret = classifier.predict(svmTrainMat);
	//�˴�����ʱ�����ţ�֮��д����Ӧ��
	cout << "ͼƬ������ɣ�ͼƬΪ��"<<ret << endl;
}

/*
��־�Ƶļ����ʶ��ʹ����ɫ���ж�λ��SVM����
1��ͼ��Ԥ����RGBתHSV
2��ͨ�����Ͷ�λ
3��SVM���ࡣ
4����ʶ�������
*/
void RGB2HSV_SVM(int imgNo, Picture_Tool picturetool) {

	char path[512];
	for (int k = 1; k <= imgNo; k++)//kΪ����ͼƬ����  
	{
		sprintf_s(path, "F:/picture/test/test%d.jpg", k);
		cout << "�����" << k << "��ͼƬ" << endl;
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
				picturetool.RGB2HSV(R, G, B, H, S, V);
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
		picturetool.FillHole(matRgb, matRgb);//���   
		//imshow("fillHole", Mat_rgb);  
		matRgb.copyTo(Mat_rgb_copy);
		vector<vector<Point> > contours;//���� 
		vector<Vec4i> hierarchy;//�ֲ�  
		findContours(matRgb, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		// ����αƽ����� + ��ȡ���κ�Բ�α߽��  
		cout << "contours: " << contours.size() << endl;
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
			/*bool iscircle = picturetool.isCircle(roiImage, temp);
			cout << "circle:" << iscircle << endl;
			if (!iscircle)
				continue;*/
			//����������״���ƣ�������Բ�α�־��
			//float C = (4 * PI*dConArea) / (dConLen*dConLen);
			//if (C < 0.5)//����Բ�ȳ�������״����ɸѡ
			//	continue;
			//copy(rect).copyTo(roiImage);
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
			putText(src, labelname[result], cvPoint(boundRect[i].x, boundRect[i].y - 10), 1, 1, CV_RGB(255, 0, 0), 2);//��ɫ����ע��  
		    //����
			sprintf_s(path, "F:/picture/test/Test_Result/%d_%d.jpg", k, count++);
			imwrite(path, src);//�������յļ��ʶ����
		}
		cout << "�����" << k << "��ͼƬ���" << endl;
	}
	cout << "�������" << endl;
	//system("pause");
	waitKey(0);
}


//����ѵ���õ�svm������
void load_svm()
{
	cout << "����SVMѵ�����" << endl;
	classifier.load(SVM_PATH);//·�� 
	cout << "�������" << endl;
}


IplImage *g_pGrayImage = NULL;
IplImage *g_pBinaryImage = NULL;
char *pstrWindowsBinaryTitle = "��ֵͼ";
void on_trackbar(int pos)
{
	// תΪ��ֵͼ  
	cvThreshold(g_pGrayImage, g_pBinaryImage, pos, 255, CV_THRESH_BINARY);
	// ��ʾ��ֵͼ  
	cvShowImage(pstrWindowsBinaryTitle, g_pBinaryImage);
	
}
void test(char* path){
	
	char *pstrWindowsToolBarName = "��ֵͼ��ֵ";

	// ���ļ��м���ԭͼ  
	IplImage *pSrcImage = cvLoadImage(path, CV_LOAD_IMAGE_UNCHANGED);

	// תΪ�Ҷ�ͼ  
	g_pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
	cvCvtColor(pSrcImage, g_pGrayImage, CV_BGR2GRAY);

	// ������ֵͼ  
	g_pBinaryImage = cvCreateImage(cvGetSize(g_pGrayImage), IPL_DEPTH_8U, 1);

	
	// ������ֵͼ����  
	cvNamedWindow(pstrWindowsBinaryTitle, CV_WINDOW_AUTOSIZE);

	// ������    
	int nThreshold = 0;
	cvCreateTrackbar(pstrWindowsToolBarName, pstrWindowsBinaryTitle, &nThreshold, 254, on_trackbar);

	on_trackbar(1);

	cvWaitKey(0);

	cvDestroyWindow(pstrWindowsBinaryTitle);
	cvReleaseImage(&pSrcImage);
	cvReleaseImage(&g_pGrayImage);
	cvReleaseImage(&g_pBinaryImage);
}
int main()
{
	//load_svm();
	//�����������ʱ��
	clock_t start, finish;
	double totaltime;
	start = clock();
	//VideoTool videotool;
	Picture_Tool picturetool;
	//ѵ��HOG+SVM
	//char *train_path = Train_PATH;
	//char *result_path = SVM_PATH;
	//Train_SVM(train_path, result_path);

	char *path = "F:/picture/test/test5.jpg";
	//saveNRGB(path);
	IplImage *testImg;
	testImg = cvLoadImage(path, CV_LOAD_IMAGE_UNCHANGED);
	cvShowImage("hahah", testImg);
	//IplImage *img = picturetool.twoValueImage(testImg, 0.12);
	//char *path = VIDEO_PATH;
	//int count = videotool.playVideo(path, picturetool);
	//cout << "count " << count << endl;
	//RGB2HSV_SVM(count, picturetool);
	//cvShowImage("erzhitu",img);
	//ʶ����ͼƬ
	//HOG_SVM_Recog(path);
	waitKey(0);
	//test(path);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "\n�˳��������ʱ��Ϊ" << totaltime << "�룡" << endl;
	return 0;
}
