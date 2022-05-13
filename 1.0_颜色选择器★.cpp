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

	int hmin = 0, smin = 0, vmin = 0;		//�����ݸ�������ֵ���и���
	int hmax = 180, smax = 255, vmax = 46;//	(Ĭ�ϳ�ɫ)

	namedWindow("�켣��", (640, 200));//�����켣��

	createTrackbar("H min", "�켣��", &hmin, 179);//����������(������)
	createTrackbar("H max", "�켣��", &hmax, 179);//����������
	createTrackbar("S min", "�켣��", &smin, 255);//����������
	createTrackbar("S max", "�켣��", &smax, 255);//����������
	createTrackbar("V min", "�켣��", &vmin, 255);//����������
	createTrackbar("V max", "�켣��", &vmax, 255);//����������


	while (1)
	{
		//cap.read(img);
		///img = imread("D://pp//yy.png");
		cvtColor(img, imgHSV, COLOR_BGR2HSV);
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		inRange(imgHSV, lower, upper, mask);//��ȡ��ɫ��Χ��������ֺ�ͼ���


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

	
	//�����ҵ�����Ҫ����ֵ

	//while (true)
	//{
	//	
	//}



}