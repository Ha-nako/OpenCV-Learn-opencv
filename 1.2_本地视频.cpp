#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/�ҵ��ļ�/��Ƶ/�����MV/�त��.mp4";//����·��
	
	VideoCapture cap(path);//���ļ�
	Mat img;

	while (true)
	{
		cap.read(img);	//��ȡΪһ֡֡��ͼ��
		namedWindow("�त��", WINDOW_FREERATIO);//�ɵ��ڴ���
		imshow("�त��", img);		
		waitKey(20);		//���20ms

	}
}