#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/我的文件/视频/混剪、MV/青い栞.mp4";//访问路径
	
	VideoCapture cap(path);//读文件
	Mat img;

	while (true)
	{
		cap.read(img);	//读取为一帧帧的图像
		namedWindow("青い栞", WINDOW_FREERATIO);//可调节窗口
		imshow("青い栞", img);		
		waitKey(20);		//间隔20ms

	}
}