#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;


void main() {

	string path = "D:/�ҵ��ļ�/ͼƬ/�����ز�/����.jpg";
	Mat img = imread(path);
	Mat imgHSV, mask;

	int hmin = 0, smin = 106, vmin = 0;		//�����ݸ�������ֵ���и���
	int hmax = 8, smax = 255, vmax = 255;

	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	namedWindow("�켣��", (640, 200));//�����켣��
	namedWindow("Image", WINDOW_FREERATIO);
	namedWindow("HSV", WINDOW_FREERATIO);
	namedWindow("mask", WINDOW_FREERATIO);

	createTrackbar("H min", "�켣��", &hmin, 179);//����������(������)
	createTrackbar("H max", "�켣��", &hmax, 179);//����������
	createTrackbar("S min", "�켣��", &smin, 255);//����������
	createTrackbar("S max", "�켣��", &smax, 255);//����������
	createTrackbar("V min", "�켣��", &vmin, 255);//����������
	createTrackbar("V max", "�켣��", &vmax, 255);//����������
	//�����ҵ�����Ҫ����ֵ
	
	while (true)
	{
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		inRange(imgHSV, lower, upper, mask);//��ȡ��ɫ��Χ��������ֺ�ͼ���

		imshow("Image", img);
		imshow("HSV", imgHSV);
		imshow("mask", mask);

		waitKey(1);
	}



}