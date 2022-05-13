#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	VideoCapture cap(0);//	摄像机ID,只有一个填0
	Mat img;

	while (true)
	{
		cap.read(img);	//读取为一帧帧的图像	或 cap >> img ;

		flip(img, img, 1);//可以实现图像反转，参数（输入，输出，参数（1为y轴反转，0为x轴反转，负数为x,y反转））
		
		imshow("video", img);
		waitKey(1);		//间隔1ms,防止播放延迟卡顿
	}
}