#include <iostream> 
#include<cstring> 
#include <stdio.h>
#include <fstream>
#include "highgui.h"
#include "cv.h"

void onTrackbarSlide(int pos);
int getAVIFrames(char * fname);
int playVideo(char *path);