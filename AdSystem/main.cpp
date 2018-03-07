#include "Picture_Tool.h"
#include "Video_Tool.h"
#include "ini_File.h"

#define VIDEO_PATH "F:/picture/AVI/WKA00925_1.mp4"
#define SVM_PATH "F:/picture/Round_Sign/train.xml"
#define Train_PATH "F:/picture/Round_Sign/train.txt"
#define Test_PATH "F:/picture/Round_Sign/Test/train.txt"
string labelname[500] = { "Around", "Forb_Drin", "Forb_Left", "Forb_Park", "Forb_right", "FororLeft", "Fororright", "Forward", "Left", "Right", "Splim100", "Splim120", "Splim20", "Splim30", "Splim50", "Splim60", "Splim70", "Splim80", "Turn_Left", "Turn_Right" };
RNG rng(12345);
CvSVM classifier;//分类器声明

/*
训练SVM分类器。
*/
void Train_SVM(char *train_path, char *result_path)
{
	int imgWidht = 48;//重新定义图片大小48*48  
	int imgHeight = 48;

	vector<string> imgTrainPath;//输入文件名变量     
	vector<int> imgTrainLabel;
	int nLine = 0;
	string buf;
	ifstream imagePath(train_path);//训练数据位置  
	unsigned long n;

	while (imagePath)//读取训练样本，imageName.txt一行为路径，一行为标签，循环      
	{
		if (getline(imagePath, buf))
		{
			nLine++;
			if (nLine % 2 == 0)
			{
				imgTrainLabel.push_back(atoi(buf.c_str()));//atoi将字符串转换成整型，图片标签      
			}
			else
			{
				imgTrainPath.push_back(buf);//图像路径      
			}
		}
	}
	imagePath.close();

	CvMat *imgDataMat, *imgLabelMat;
	int nImgNum = nLine / 2; //因为是每隔一行才是图片路径，所以要除以2         
	imgDataMat = cvCreateMat(nImgNum, 900, CV_32FC1);  //cmd命令行运行出的HOG dims （即descriptors.size()）大小，不同图片数值不一样，请自行修改    
	cvSetZero(imgDataMat);
	imgLabelMat = cvCreateMat(nImgNum, 1, CV_32FC1);
	cvSetZero(imgLabelMat);

	IplImage* srcImg;
	IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
	cout << "HOG特征开始提取" << endl;
	for (string::size_type i = 0; i != imgTrainPath.size(); i++)
	{
		srcImg = cvLoadImage(imgTrainPath[i].c_str(), 1);
		if (srcImg == NULL)
		{
			cout << " 图片读取错误 " << imgTrainPath[i].c_str() << endl;
			continue;
		}
		cvResize(srcImg, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float>descriptors;//数组的结果         
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //开始计算         
		cout << "HOG dims（descriptors.size()）: " << descriptors.size() << endl;
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(imgDataMat, i, n, *iter);//HOG存储      
			n++;
		}
		cvmSet(imgLabelMat, i, 0, imgTrainLabel[i]);
	}
	cout << "HOG特征结束提取" << endl;

	CvSVM svm;//新建SVM   
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);//参数  
	cout << "svm开始训练" << endl;
	clock_t start = clock();
	svm.train(imgDataMat, imgLabelMat, NULL, NULL, param);//训练svm      
	clock_t finish = clock();
	double consumeTime = (double)(finish - start);
	cout << "svm训练结束,用时" << consumeTime << endl;
	svm.save(result_path);//训练结果存储位置  
	cvReleaseMat(&imgDataMat);
	cvReleaseMat(&imgLabelMat);

	system("pause");
	//return 0;
}
//载入训练好的svm分类器
void load_svm()
{
	cout << "导入SVM训练结果" << endl;
	classifier.load(SVM_PATH);//路径 
	cout << "导入完成" << endl;
}

/**
使用HOG特征与SVM进行分类并计算准确率，此函数的作用是检测特征与分类的效果。
1、根据图片的路径和标签来分类图片,
2、并将结果保存到predictResult.txt文件中，
3、并计算分类成功率
*/
void HOG_SVM_MulRecog(char *path)
{
	int imgWidht = 48;//重新定义图片大小48*48  
	int imgHeight = 48;
	IplImage *testImg;
	vector<string> testImgPath;//测试图片的路径  
	vector<int> realTestImgLabel;
	vector<int> predictTestImgLabel;
	int predictRightNum = 0;
	double predictRightRatio;
	ifstream readTestImgPath(path);//读取测试图片路径，txt文件为图片路径名称  
	string buf;
	int nLine = 0;
	while (readTestImgPath)
	{
		if (getline(readTestImgPath, buf))
		{
			nLine++;
			if (nLine % 2 == 0)
				realTestImgLabel.push_back(atoi(buf.c_str()));//存标签  
			else
				testImgPath.push_back(buf);//存路径  
		}
	}
	readTestImgPath.close();
	char line[512];
	ofstream predictResultPath("F:/picture/Round_Sign/Test/predictResultRandom.txt");//预测结果存储在此文本     
	for (int j = 0; j < testImgPath.size(); j++)//读取测试图片      
	{
		testImg = cvLoadImage(testImgPath[j].c_str(), 1);
		if (testImg == NULL)
		{
			cout << "图片读取错误 " << testImgPath[j].c_str() << endl;
			continue;
		}

		IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
		cvZero(trainImg);
		cvResize(testImg, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);//hog特征训练  
		vector<float> descriptors;//数组的结果         
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //开始计算         
		//cout << "HOG dims: " << descriptors.size() << endl;
		CvMat* svmTrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
		unsigned long n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(svmTrainMat, 0, n, *iter);
			n++;
		}
		//预测
		int ret = classifier.predict(svmTrainMat);
		sprintf(line, "%s %d\r\n", testImgPath[j].c_str(), ret);
		predictResultPath << line;

		if (hog != NULL)
		{
			delete hog;
			hog = NULL;
		}
	}
	predictResultPath.close();
	for (int i = 0; i < realTestImgLabel.size() && i<predictTestImgLabel.size(); i++)
	{
		if (realTestImgLabel[i] == predictTestImgLabel[i])//判断实际值与预测值是否相等  
			predictRightNum++;
	}
	predictRightRatio = (double)predictRightNum / predictTestImgLabel.size();//计算预测正确的比例  
	cout << "一共预测" << predictTestImgLabel.size() << "张图片," << "正确率为：" << predictRightRatio << endl;
}

/*
	对一张图片进行识别
*/
void HOG_SVM_SinRecog(char *img_path)
{
	int imgWidht = 48;//重新定义图片大小48*48  
	int imgHeight = 48;
	IplImage *testImg;
	testImg = cvLoadImage(img_path, 1);
	if (testImg == NULL)
	{
		cout << "图片读取错误 " << img_path << endl;
		exit(0);
	}

	IplImage* trainImg = cvCreateImage(cvSize(imgWidht, imgHeight), 8, 3);
	cvZero(trainImg);
	cvResize(testImg, trainImg);
	//提取HOG特征
	HOGDescriptor *hog = new HOGDescriptor(cvSize(imgWidht, imgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);//hog特征训练  
	vector<float> descriptors;//数组的结果         
	hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //开始计算         
	cout << "HOG dims: " << descriptors.size() << endl;
	CvMat* svmTrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
	unsigned long n = 0;
	for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		cvmSet(svmTrainMat, 0, n, *iter);
		n++;
	}
	//对图片进行预测
	int ret = classifier.predict(svmTrainMat);
	//输出名称
	cout << "图片处理完成，图片为：" << labelname[ret] << endl;
}

/*
标志牌的检测与识别，使用颜色进行定位，SVM分类
1、图像预处理，RGB转HSV
2、通过膨胀定位
3、SVM分类。
4、将识别结果标出
*/
void RGB2HSV_SVM(int imgNo, Picture_Tool picturetool) {

	char path[512];
	for (int k = 1; k <= imgNo; k++)//k为测试图片数量  
	{
		sprintf_s(path, "F:/picture/test/test%d.jpg", k);
		cout << "分类第" << k << "张图片" << endl;
		Mat src = imread(path);
		Mat copy;
		src.copyTo(copy);
		int width = src.cols;   //图像宽度  
		int height = src.rows;   //图像高度  
		//色彩分割  
		double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
		Mat matRgb = Mat::zeros(src.size(), CV_8UC1);
		Mat Mat_rgb_copy;//一个暂存单元  
		int x, y;
		for (y = 0; y < height; y++)
		{
			for (x = 0; x<width; x++)
			{
				B = src.at<Vec3b>(y, x)[0];
				G = src.at<Vec3b>(y, x)[1];
				R = src.at<Vec3b>(y, x)[2];
				picturetool.RGB2HSV(R, G, B, H, S, V);
				//红色：337-360  
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
		erode(matRgb, matRgb, element);//腐蚀  
		//imshow("erode", Mat_rgb);  
		dilate(matRgb, matRgb, element1);//膨胀  
		//imshow("dilate", Mat_rgb);  
		picturetool.FillHole(matRgb, matRgb);//填充   
		//imshow("fillHole", Mat_rgb);  
		matRgb.copyTo(Mat_rgb_copy);
		vector<vector<Point> > contours;//轮廓 
		vector<Vec4i> hierarchy;//分层  
		findContours(matRgb, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		// 多边形逼近轮廓 + 获取矩形和圆形边界框  
		cout << "contours: " << contours.size() << endl;
		vector<vector<Point> > contours_poly(contours.size());//近似后的轮廓点集   
		vector<Rect> boundRect(contours.size()); //包围点集的最小矩形vector    
		vector<Point2f>center(contours.size());//包围点集的最小圆形vector   
		vector<float>radius(contours.size());//包围点集的最小圆形半径vector   

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//对多边形曲线做适当近似，contours_poly[i]是输出的近似点集  
			boundRect[i] = boundingRect(Mat(contours_poly[i]));//计算并返回包围轮廓点集的最小矩形   
			minEnclosingCircle(contours_poly[i], center[i], radius[i]);//计算并返回包围轮廓点集的最小圆形及其半径  
		}
		Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
		int count = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundRect[i];
			//cout << rect<<endl;  
			//高宽比限制  
			float ratio = (float)rect.width / (float)rect.height;
			//轮廓面积       
			float Area = (float)rect.width * (float)rect.height;
			float dConArea = (float)contourArea(contours[i]);
			float dConLen = (float)arcLength(contours[i], 1);
			if (dConArea <400)
				continue;
			if (ratio>2 || ratio < 0.5)
				continue;

			//进行圆筛选，通过四块的缺失像素比较  
			Mat roiImage;
			Mat_rgb_copy(rect).copyTo(roiImage);
			//imshow("roiImage",roiImage);  
			//imshow("test",roiImage);  
			Mat temp;
			copy(rect).copyTo(temp);
			//imshow("test2",temp);//显示从场景图中提取出的标识，留着。  
			/*bool iscircle = picturetool.isCircle(roiImage, temp);
			cout << "circle:" << iscircle << endl;
			if (!iscircle)
				continue;*/
			//接下来是形状限制！这里检测圆形标志牌
			//float C = (4 * PI*dConArea) / (dConLen*dConLen);
			//if (C < 0.5)//利用圆度初步对形状进行筛选
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

			int result = (int)classifier.predict(temp2) - 1;//svm预测  
			Scalar color = (0, 0, 255);//蓝色线画轮廓   
			drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			putText(src, labelname[result], cvPoint(boundRect[i].x, boundRect[i].y - 10), 1, 1, CV_RGB(255, 0, 0), 2);//红色字体注释  
		    //保存
			sprintf_s(path, "F:/picture/test/Test_Result/%d_%d.jpg", k, count++);
			imwrite(path, src);//保存最终的检测识别结果
		}
		cout << "分类第" << k << "张图片完成" << endl;
	}
	cout << "分类完成" << endl;
	waitKey(0);
}
void Hough(Mat srcBw){

	Mat edges;  //义转化的灰度图
	namedWindow("效果图", CV_WINDOW_NORMAL);
	
	//cvtColor(srcBw, edges, CV_BGR2GRAY);
	//高斯滤波
	GaussianBlur(edges, edges, Size(7, 7), 2, 2);
	vector<Vec3f> circles;
	
	//霍夫圆
	HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
	//	//绘制圆心  
		circle(srcBw, center, 3, Scalar(0, 255, 0), -1, 8, 0);
	//	//绘制圆轮廓  
		circle(srcBw, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}
	imshow("效果图", srcBw);

	//waitKey(0);
}
/*
标志牌的检测与识别，使用颜色进行定位，SVM分类
1、图像预处理，RGB转HSV
2、通过膨胀定位
3、SVM分类。
4、将识别结果标出
*/
void RGB2HSV_SVM_ONE(char* imgpath, Picture_Tool picturetool) {

	char path[512];
	Mat src = imread(imgpath);
	Mat copy;
	src.copyTo(copy);
	int width = src.cols;   //图像宽度  
	int height = src.rows;   //图像高度  
	//色彩分割  
	double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
	Mat matRgb = Mat::zeros(src.size(), CV_8UC1);
	Mat Mat_rgb_copy;//一个暂存单元  
	int x, y;
	for (y = 0; y < height; y++)
	{
		for (x = 0; x<width; x++)
		{
			B = src.at<Vec3b>(y, x)[0];
			G = src.at<Vec3b>(y, x)[1];
			R = src.at<Vec3b>(y, x)[2];
			picturetool.RGB2HSV(R, G, B, H, S, V);
			//红色：337-360  
			if ((H >= 337 && H <= 360 || H >= 0 && H <= 10) && S >= 12 && S <= 100 && V>20 && V < 99)
			{
				matRgb.at<uchar>(y, x) = 255;
			}
		}
	}
	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
	//imshow("MyWindow", matRgb);

	medianBlur(matRgb, matRgb, 3);
	//imshow("MyWindow", matRgb); 
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
	erode(matRgb, matRgb, element);//腐蚀  
	//imshow("MyWindow", matRgb);
	dilate(matRgb, matRgb, element1);//膨胀  
	//imshow("MyWindow", matRgb);  
	picturetool.FillHole(matRgb, matRgb);//填充 
	Mat canSrc;
	Canny(matRgb, matRgb, 10, 250, 5);
	imshow("MyWindow", matRgb); 
	//Hough(matRgb);
	vector<Vec3f> circles;
	HoughCircles(matRgb, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 0, 0);
	cout << circles.size() << endl;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));

		int radius = cvRound(circles[i][2]);
		//绘制圆心  
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		//绘制圆轮廓  
		circle(src, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}
	imshow("圆形检测", src);
//	matRgb.copyTo(Mat_rgb_copy);
//	vector<vector<Point> > contours;//轮廓 
//	vector<Vec4i> hierarchy;//分层  
//	findContours(matRgb, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//	// 多边形逼近轮廓 + 获取矩形和圆形边界框  
//	//cout << "contours: " << contours.size() << endl;
//	vector<vector<Point> > contours_poly(contours.size());//近似后的轮廓点集   
//	vector<Rect> boundRect(contours.size()); //包围点集的最小矩形vector    
////	vector<Point2f>center(contours.size());//包围点集的最小圆形vector   
////	vector<float>radius(contours.size());//包围点集的最小圆形半径vector   
//
//	for (int i = 0; i < contours.size(); i++)
//	{
//		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//对多边形曲线做适当近似，contours_poly[i]是输出的近似点集  
//		boundRect[i] = boundingRect(Mat(contours_poly[i]));//计算并返回包围轮廓点集的最小矩形   
////		minEnclosingCircle(contours_poly[i], center[i], radius[i]);//计算并返回包围轮廓点集的最小圆形及其半径  
//	}
//	Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
//	int count = 0;
//	for (int i = 0; i < contours.size(); i++)
//	{
//		Rect rect = boundRect[i];
//		//cout << rect<<endl;  
//		//高宽比限制  
//		float ratio = (float)rect.width / (float)rect.height;
//		//轮廓面积       
//		float Area = (float)rect.width * (float)rect.height;
//		float dConArea = (float)contourArea(contours[i]);
//		float dConLen = (float)arcLength(contours[i], 1);
//		if (dConArea <400)
//			continue;
//		if (ratio>2 || ratio < 0.5)
//			continue;
//
//		//进行圆筛选，通过四块的缺失像素比较  
//		Mat roiImage;
//		Mat_rgb_copy(rect).copyTo(roiImage);
//		//imshow("MyWindow",roiImage);   
//		Mat temp;
//		copy(rect).copyTo(temp);
//		//显示从场景图中提取出的标识，留着。  
//		/*bool iscircle = picturetool.isCircle(roiImage, temp);
//		cout << "circle:" << iscircle << endl;
//		if (!iscircle)
//		continue;*/
//		//接下来是形状限制！这里检测圆形标志牌
//		//float C = (4 * PI*dConArea) / (dConLen*dConLen);
//		//if (C < 0.5)//利用圆度初步对形状进行筛选
//		//	continue;
//		//copy(rect).copyTo(roiImage);
//		//*********svm*********  
//		Mat temp2 = Mat::zeros(temp.size(), CV_8UC1);
//		cvtColor(temp, temp2, CV_BGR2GRAY);
//		//resize(temp2, temp2, Size(48, 48));  
//		resize(temp2, temp2, Size(30, 30));//30*30=900  
//		temp2 = temp2.reshape(0, 1);
//		temp2.convertTo(temp2, CV_32F);
//		//cout << temp2.size() << endl;
//
//		//int result = (int)classifier.predict(temp2) - 1;//svm预测  
//		Scalar color = (0, 0, 255);//蓝色线画轮廓   
//		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
//		rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
//		//putText(src, labelname[result], cvPoint(boundRect[i].x, boundRect[i].y - 10), 1, 1, CV_RGB(255, 0, 0), 2);//红色字体注释  
//		//保存
//		sprintf_s(path, "F:/picture/test/Test_Result/%d_%d.jpg", 0, count++);
//		imwrite(path, src);//保存最终的检测识别结果
//	}
	waitKey(0);
	destroyWindow("MyWindow");
}

int main()
{
	//load_svm();
	//计算程序运行时间
	clock_t start, finish;
	double totaltime;
	start = clock();
	VideoTool videotool;
	Picture_Tool picturetool;

	//训练HOG+SVM
	//char *train_path = Train_PATH;
	//char *result_path = SVM_PATH;
	//Train_SVM(train_path, result_path);

	char *path = "F:/picture/Tsinghua-Tencent/data/test/394.jpg";
	char *blue = "F:/picture/Tsinghua-Tencent/data/test/2105.jpg";
	char *path1 = "F:/picture/FullIJCNN2013/00002.ppm";
	
//	RGB2HSV_SVM_ONE(path, picturetool);
	//saveNRGB(path);
	IplImage *testImg,*histImg;
	testImg = cvLoadImage(blue);
	cvShowImage("原图", testImg);

	histImg = picturetool.EqualizeHistColorImage(testImg);
	//cvShowImage("均衡化", histImg);
	IplImage *img = picturetool.NormalizeImage(histImg);
	//cvShowImage("归一化后图", img);
	
	IplImage *img1 = picturetool.twoValueImage(img, 2);
	cvShowImage("RGBN", img1);
	//char *path = VIDEO_PATH;
	//int count = videotool.playVideo(path, picturetool);
	//cout << "count " << count << endl;
	//RGB2HSV_SVM(count, picturetool);
	//cvShowImage("erzhitu",img);
	//识别单张图片
	//HOG_SVM_Recog(path);
	//char *test_path = Test_PATH;
	//识别多张图片
	//HOG_SVM_MulRecog(test_path);

	waitKey(0);
	//test(path);
	
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "\n此程序的运行时间为" << totaltime << "秒！" << endl;
	return 0;
}
void Hough()
{
	int kvalue = 15;
	Mat src_color = imread("F:/picture/test/test.jpg");//读取原彩色图
	imshow("原图-彩色", src_color);

	//声明一个三通道图像，像素值全为0，用来将霍夫变换检测出的圆画在上面
	Mat dst(src_color.size(), src_color.type());
	dst = Scalar::all(0);

	Mat src_gray;//彩色图像转化成灰度图
	cvtColor(src_color, src_gray, COLOR_RGB2GRAY);
	imshow("原图-灰度", src_gray);
	//imwrite("src_gray.png", src_gray);

	Mat bf;//对灰度图像进行双边滤波
	bilateralFilter(src_gray, bf, kvalue, kvalue * 2, kvalue / 2);
	//imshow("灰度双边滤波处理", bf);
	//imwrite("src_bf.png", bf);

	Mat canSrc;//对灰度图像进行双边滤波
	Canny(bf, canSrc, 10, 250, 5);
	imshow("canny", canSrc);

	bilateralFilter(canSrc, dst, kvalue, kvalue * 2, kvalue / 2);
	vector<Vec3f> circles;//声明一个向量，保存检测出的圆的圆心坐标和半径
	//    HoughCircles(canSrc, circles, CV_HOUGH_GRADIENT, 1.5, 100, 130, 38, 20, 100);//霍夫变换检测圆
	HoughCircles(canSrc, circles, CV_HOUGH_GRADIENT, 1.5, 100, 130, 38, 20, 300);//霍夫变换检测圆

	cout << circles.size() << endl;
	cout << "x=\ty=\tr=" << endl;
	for (size_t i = 0; i < circles.size(); i++)//把霍夫变换检测出的圆画出来
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(dst, center, 0, Scalar(0, 255, 0), -1, 8, 0);
		circle(dst, center, radius, Scalar(255, 255, 255), 1, 8, 0);

		cout << cvRound(circles[i][0]) << "\t" << cvRound(circles[i][1]) << "\t"
			<< cvRound(circles[i][2]) << endl;//在控制台输出圆心坐标和半径               
	}
	//dilate(canSrc, dst,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
	imshow("特征提取", dst);
	//imwrite("dst.png", dst);

	waitKey();
}