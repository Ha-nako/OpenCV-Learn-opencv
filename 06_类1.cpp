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
	//��ֵ�˲�
	medianBlur(srcImage, srcImage, 3);
	//ת����HSV��ɫ�ռ�
	Mat hsvImage;
	cvtColor(srcImage, hsvImage, CV_BGR2HSV);
	imshow("hsv", hsvImage);
	//��ɫ��ֵ������
	//����ߵ���ֵ
	Mat lowMat,mask;
	Mat upperMat;
	Scalar lower(34, 30, 214);
	Scalar upper(142, 123, 255);

	inRange(hsvImage, lower, upper, mask);//��ȡ��ɫ��Χ��������ֺ�ͼ���
	imshow("mask", mask);

	//��˹�˲�
	GaussianBlur(mask, mask, Size(9, 9), 2, 2);
	//����Բ�Ӳ�
	vector<Vec3f>  circles;
	HoughCircles(mask, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);

	  //���û�м�⵽Բ
	if (circles.size() == 0)
		return -1;
		for (int i = 0; i < circles.size(); i++)
		{
			//���Բ�ĵ�λ�ú�Բ�뾶�Ĵ�С
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(resultImag, center, radius, Scalar(155, 50, 255), 5);
			std::cout << "Բ�İ뾶��" << radius << std::endl;
			std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
		}
	imshow("resultImag", resultImag);
	waitKey(0);
	return 0;
}



