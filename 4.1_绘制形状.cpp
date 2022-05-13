#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main()
{
	Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));
	//    CV_8U :8位(0-255)、C3 :RGB 3通道、Scalar内为BGR值
	//										↑颜色缩放器（标量）

	circle(img, Point(256, 256), 155, Scalar(0, 69, 255),FILLED);
	//			↑圆心			↑大小	↑颜色			 ↑厚度(FILLED:填满)
	
	//circle(img，圆心，大小，颜色，厚度[FILLED表示全部填充])

	rectangle(img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), FILLED);
	//				↑矩形左上角	↑右下角
	

	line(img, Point(130, 296), Point(382, 296), Scalar(255, 255, 255), 3);
	//↑	画一条线												   ↑厚度

	putText(img, "My name's Liu", Point(137, 262), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255),2);
	//文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑


	imshow("Image", img);

	waitKey(0);
}