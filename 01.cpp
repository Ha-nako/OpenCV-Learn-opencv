#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;

void SendMessageOne()
{
	//开起摄像头
	VideoCapture capture;
	capture.open(0);
	Mat edges;  //定义转化的灰度图
	if (!capture.isOpened())
		namedWindow("【效果图】", CV_WINDOW_NORMAL);

	while (1)
	{
		int Y = 0, J = 0;
		Mat frame;
		capture >> frame;
		cvtColor(frame, edges, CV_BGR2GRAY);
		//高斯滤波
		GaussianBlur(edges, edges, Size(7, 7), 2, 2);
		std::vector<Vec3f> circles;//存储每个圆的位置信息
		//霍夫圆
		HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 1.5, 10, 100, 100, 0, 100);
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);

			//绘制圆轮廓  
			circle(frame, center, radius, Scalar(155, 50, 255), 3, 8, 0);
			int R = frame.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
			int G = frame.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
			int B = frame.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
			std::cout << "圆的半径是" << radius << std::endl;
			std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;
		}

		imshow("【效果图】", frame);
		waitKey(30);
	}
}

int main()
{
	std::thread* a = new std::thread(SendMessageOne);
	a->join();
	return 0;
}