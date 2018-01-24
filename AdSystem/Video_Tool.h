#include <iostream> 
#include<cstring> 
#include <stdio.h>
#include <fstream>
#include "highgui.h"
#include "cv.h"
#include "Picture_Tool.h"
class VideoTool {
private:
	void onTrackbarSlide(int pos);
	int getAVIFrames(char * fname);
public:
	int playVideo(char *path,Picture_Tool picturetool);
};