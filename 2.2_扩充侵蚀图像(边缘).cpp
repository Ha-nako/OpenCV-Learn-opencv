#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	//↓访问图片路径
	string path = "D:/A_file/1.jpg";
	Mat img = imread(path);	//读文件
	Mat imgblur, imgCanny,imgDil,imgErode;	
	Mat imgs;
	VideoCapture cap(0);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//↑高斯模糊:size(定义内核大小7*7),输出屏幕上的位置偏移->5,0(x,y)

	Canny(imgblur, imgCanny, 50, 150);
	//↑坎尼边沿检测	  两个检测阈值(可更改,值越小边缘越多)

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	//↑定义可使用膨胀(扩充/侵蚀)的内核  (数越小,扩充越多,只能用奇数)

	dilate(imgCanny, imgDil, kernel);	
	//↑	边缘扩充

	erode(imgDil, imgErode, kernel);
	//↑	侵蚀
	

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	namedWindow("图片", WINDOW_FREERATIO);
	namedWindow("高斯模糊", WINDOW_FREERATIO);
	namedWindow("坎尼边沿检测", WINDOW_FREERATIO);
	namedWindow("边缘扩充", WINDOW_FREERATIO);
	namedWindow("边缘侵蚀", WINDOW_FREERATIO);
	//namedWindow("网络摄像头边缘检测", WINDOW_FREERATIO);

	imshow("图片", img);
	imshow("高斯模糊", imgblur);
	imshow("坎尼边沿检测", imgCanny); 
	imshow("边缘扩充", imgDil);
	imshow("边缘侵蚀", imgErode);

	while (true)
	{
		cap.read(imgs);
		Canny(imgs, imgCanny, 50, 150);
		imshow("网络摄像头边缘检测", imgCanny);
		waitKey(1);
	}


	waitKey(0);		//延迟显示
}