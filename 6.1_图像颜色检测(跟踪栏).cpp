#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;


void main() {

	string path = "D:/我的文件/图片/所用素材/汽车.jpg";
	Mat img = imread(path);
	Mat imgHSV, mask;

	int hmin = 0, smin = 106, vmin = 0;		//随后根据跟踪栏数值进行更改
	int hmax = 8, smax = 255, vmax = 255;

	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	namedWindow("轨迹栏", (640, 200));//创建轨迹栏
	namedWindow("Image", WINDOW_FREERATIO);
	namedWindow("HSV", WINDOW_FREERATIO);
	namedWindow("mask", WINDOW_FREERATIO);

	createTrackbar("H min", "轨迹栏", &hmin, 179);//创建跟踪栏(滑动条)
	createTrackbar("H max", "轨迹栏", &hmax, 179);//创建跟踪栏
	createTrackbar("S min", "轨迹栏", &smin, 255);//创建跟踪栏
	createTrackbar("S max", "轨迹栏", &smax, 255);//创建跟踪栏
	createTrackbar("V min", "轨迹栏", &vmin, 255);//创建跟踪栏
	createTrackbar("V max", "轨迹栏", &vmax, 255);//创建跟踪栏
	//更快找到所需要的数值
	
	while (true)
	{
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		inRange(imgHSV, lower, upper, mask);//提取颜色范围并输出遮罩后图像★

		imshow("Image", img);
		imshow("HSV", imgHSV);
		imshow("mask", mask);

		waitKey(1);
	}



}