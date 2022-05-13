#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;


void main() {

	VideoCapture cap(0);
	Mat img ;
	Mat imgHSV, mask;

	int hmin = 0, smin = 0, vmin = 0;		//随后根据跟踪栏数值进行更改
	int hmax = 180, smax = 255, vmax = 46;//	(默认橙色)

	namedWindow("轨迹栏", (640, 200));//创建轨迹栏

	createTrackbar("H min", "轨迹栏", &hmin, 179);//创建跟踪栏(滑动条)
	createTrackbar("H max", "轨迹栏", &hmax, 179);//创建跟踪栏
	createTrackbar("S min", "轨迹栏", &smin, 255);//创建跟踪栏
	createTrackbar("S max", "轨迹栏", &smax, 255);//创建跟踪栏
	createTrackbar("V min", "轨迹栏", &vmin, 255);//创建跟踪栏
	createTrackbar("V max", "轨迹栏", &vmax, 255);//创建跟踪栏


	while (1)
	{
		//cap.read(img);
		///img = imread("D://pp//yy.png");
		cvtColor(img, imgHSV, COLOR_BGR2HSV);
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		inRange(imgHSV, lower, upper, mask);//提取颜色范围并输出遮罩后图像★


		imshow("Image", img);
		imshow("HSV", imgHSV);
		imshow("mask", mask);
		//system("color a");
		cout << hmin << "," << smin << "," << vmin << "," << hmax << "," << smax << "," << vmax << endl;
		waitKey(1);

	}
	

	
	//namedWindow("Image", WINDOW_FREERATIO);
	//namedWindow("HSV", WINDOW_FREERATIO);
	//namedWindow("mask", WINDOW_FREERATIO);

	
	//更快找到所需要的数值

	//while (true)
	//{
	//	
	//}



}