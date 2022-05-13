#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main()
{
	Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));
	//    CV_8U :8λ(0-255)��C3 :RGB 3ͨ����Scalar��ΪBGRֵ
	//										����ɫ��������������

	circle(img, Point(256, 256), 155, Scalar(0, 69, 255),FILLED);
	//			��Բ��			����С	����ɫ			 �����(FILLED:����)
	
	//circle(img��Բ�ģ���С����ɫ�����[FILLED��ʾȫ�����])

	rectangle(img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), FILLED);
	//				���������Ͻ�	�����½�
	

	line(img, Point(130, 296), Point(382, 296), Scalar(255, 255, 255), 3);
	//��	��һ����												   �����

	putText(img, "My name's Liu", Point(137, 262), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255),2);
	//���֣�	  ������					       �����壨�����		������		      ��ȡ�


	imshow("Image", img);

	waitKey(0);
}