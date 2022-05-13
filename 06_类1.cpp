#include <opencv2/imgproc/imgproc.hpp> 
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector> 
#include <math.h>

using namespace cv;
using namespace std;

int main()
{
	Mat srcImage = imread("D://pp//yy.png");
	if (!srcImage.data)
	{
		printf("could not load image...\n");
		return -1;
	}
	imshow("srcImage", srcImage);
	Mat resultImag = srcImage.clone();
	//中值滤波
	medianBlur(srcImage, srcImage, 3);
	//转换成HSV颜色空间
	Mat hsvImage;
	cvtColor(srcImage, hsvImage, CV_BGR2HSV);
	imshow("hsv", hsvImage);
	//颜色阈值化处理
	//定义高低阈值
	Mat lowMat,mask;
	Mat upperMat;
	Scalar lower(34, 30, 214);
	Scalar upper(142, 123, 255);

	inRange(hsvImage, lower, upper, mask);//提取颜色范围并输出遮罩后图像★
	imshow("mask", mask);

	//高斯滤波
	GaussianBlur(mask, mask, Size(9, 9), 2, 2);
	//霍夫圆加测
	vector<Vec3f>  circles;
	HoughCircles(mask, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);

	  //如果没有检测到圆
	if (circles.size() == 0)
		return -1;
		for (int i = 0; i < circles.size(); i++)
		{
			//求出圆心的位置和圆半径的大小
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(resultImag, center, radius, Scalar(155, 50, 255), 5);
			std::cout << "圆的半径是" << radius << std::endl;
			std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;
		}
	imshow("resultImag", resultImag);
	waitKey(0);
	return 0;
}



