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
	Mat lowMat;
	Mat upperMat;
	//红色H的范围是[0,10]  [160,180]
	inRange(hsvImage, Scalar(34, 30, 214), Scalar(34, 123, 255), lowMat);//红色
	inRange(hsvImage, Scalar(34, 30, 214), Scalar(34, 123, 255), upperMat);
	imshow("lowMat", lowMat);
	imshow("upperMat", upperMat);
	//将高低阈值合并
	Mat redMat;
	addWeighted(lowMat, 1, upperMat, 1, 0, redMat);
	imshow("redMat", redMat);
	//高斯滤波
	GaussianBlur(redMat, redMat, Size(9, 9), 2, 2);
	//霍夫圆加测
	vector<Vec3f>  circles;
	HoughCircles(redMat, circles, CV_HOUGH_GRADIENT, 1, redMat.rows / 8, 100, 20, 0, 0);
  //HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//如果没有检测到圆
	if (circles.size() == 0)
		//return -1;
	for (int i = 0; i < circles.size(); i++)
	{
		//求出圆心的位置和圆半径的大小
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(resultImag, center, radius, Scalar(0, 255, 0), 5);
	}
	imshow("resultImag", resultImag);
	waitKey(0);
	return 0;
}




//string path = "D://pp//yy.png";
	//Mat img = imread(path);	//读文件
	//Mat imggray, imgblur, imgCanny, imgDil;	//定义图像变量


	//cvtColor(img, imggray, COLOR_BGR2GRAY);
	////↑转化灰度图像:1转化为2,模式为3（灰度）

	//GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	////↑高斯模糊:size(定义内核大小7*7),输出屏幕上的位置偏移->5,0(x,y)

	//Canny(imgblur, imgCanny, 30, 100);
	////↑坎尼边沿检测	  两个检测阈值(可更改,值越小边缘越多)

	//namedWindow("原图", WINDOW_FREERATIO);
	////namedWindow("坎尼边沿检测", WINDOW_FREERATIO);

	//imshow("原图", img);
	////imshow("坎尼边沿检测", imgCanny);

	//Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	////↑定义可使用膨胀(扩充/侵蚀)的内核  (数越小,扩充越多,只能用奇数)
	//dilate(imgCanny, imgDil, kernel);
	////↑	边缘扩充
	////imshow("边缘扩充", imgDil);



	//cv2.kmeans(
	//	InputArray data,
	//	int K,
	//	InputOutputArray bestLabels,
	//	TermCriteria criteria,
	//	int attempts,
	//	int flags,
	//	OutputArray centers = noArray()
	//)



	//imshow("【效果图】", img);


	//waitKey(0);		//延迟显示
	// 
	// -----------------------------------------------------------------------------
//-----霍夫圆-----
	//std::vector<Vec3f> circles;//存储每个圆的位置信息
	//	//霍夫圆
	//HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	////																		
	//for (size_t i = 0; i < circles.size(); i++)
	//{
	//	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	//	int radius = cvRound(circles[i][2]);

	//	//绘制圆轮廓  
	//	circle(img, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	//	int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
	//	int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
	//	int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
	//	std::cout << "圆的半径是" << radius << std::endl;
	//	std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;
	//}