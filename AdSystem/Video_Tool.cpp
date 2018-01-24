#include "Video_Tool.h"
#include "Picture_Tool.h"

using namespace std;

int        g_slider_position = 0;
CvCapture* g_capture = NULL;


void VideoTool::onTrackbarSlide(int pos) {
	cvSetCaptureProperty(
		g_capture,
		CV_CAP_PROP_POS_FRAMES,
		pos
		);
}

//Hack because sometimes the number of frames in a video is not accessible. 
//Probably delete this on Widows
int VideoTool::getAVIFrames(char * fname) {
	char tempSize[4];
	// Trying to open the video file
	ifstream  videoFile(fname, ios::in | ios::binary);
	// Checking the availablity of the file
	if (!videoFile) {
		cout << "Couldn’t open the input file " << fname << endl;
		exit(1);
	}
	// get the number of frames
	videoFile.seekg(0x30, ios::beg);
	videoFile.read(tempSize, 4);
	int frames = (unsigned char)tempSize[0] + 0x100 * (unsigned char)tempSize[1] + 0x10000 * (unsigned char)tempSize[2] + 0x1000000 * (unsigned char)tempSize[3];
	videoFile.close();
	return frames;
}


int VideoTool::playVideo(char *path,Picture_Tool picturetool) {

	cvNamedWindow("Test", CV_WINDOW_AUTOSIZE);
	CvCapture* g_capture = cvCreateFileCapture(path);
	IplImage *foo = cvQueryFrame(g_capture);
	int frames = (int)cvGetCaptureProperty(
		g_capture,
		CV_CAP_PROP_FRAME_COUNT
		);

	int tmpw = (int)cvGetCaptureProperty(
		g_capture,
		CV_CAP_PROP_FRAME_WIDTH
		);

	int tmph = (int)cvGetCaptureProperty(
		g_capture,
		CV_CAP_PROP_FRAME_HEIGHT
		);

	printf("opencv frames %d w %d h %d\n", frames, tmpw, tmph);

	//获取视频fps
	/*int fps = (int)cvGetCaptureProperty(g_capture, CV_CAP_PROP_FPS);
	cout << "fps:" << fps << endl;*/

	frames = getAVIFrames(path); //This is a hack because on linux, getting number of frames often doesn't work

	printf("hacked frames %d w %d h %d\n", frames, tmpw, tmph);

	IplImage* frame;
	frames = 0;
	int count = 0;
	char tpath[512];
	cout << "开始处理" << endl;
	while (1) {
		frame = cvQueryFrame(g_capture);
		if (!frame) break;
		//      int frames = cvGetCaptureProperty( g_capture, CV_CAP_PROP_POS_FRAMES);//This should work, sometimes it does not on linux
		//printf("\nFrame number=%d\n", frames);
		//cvSetTrackbarPos("Position", "Example2_3", frames);
		frames++;
		if (frames % 20 == 0){
			// 将帧转成图片输出  
			cout << "保存第" <<frames<<"帧"<< endl;
			sprintf_s(tpath, "F:/picture/test/test%d.jpg", ++count);
			//cvSaveImage(tpath, picturetool.NormalizeImage(frame));
			cvSaveImage(tpath, frame);
			//cout << "开始分类第" << frames << "帧" << endl;
			//RGB2HSV_SVM(tpath);
			//cout << "第" << frames << "分类完成帧" << endl;
		}
		cvShowImage("Test", frame);
		char c = (char)cvWaitKey(10);
		if (c == 27) break;
	}
	cvReleaseCapture(&g_capture);
	cvDestroyWindow("Test");
	return count;
}