#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	//������ͼƬ·��
	string path = "D:/A_file/1.jpg";
	Mat img = imread(path);	//���ļ�
	Mat imgblur, imgCanny,imgDil,imgErode;	
	Mat imgs;
	VideoCapture cap(0);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)

	Canny(imgblur, imgCanny, 50, 150);
	//��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��,ֻ��������)

	dilate(imgCanny, imgDil, kernel);	
	//��	��Ե����

	erode(imgDil, imgErode, kernel);
	//��	��ʴ
	

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	namedWindow("ͼƬ", WINDOW_FREERATIO);
	namedWindow("��˹ģ��", WINDOW_FREERATIO);
	namedWindow("������ؼ��", WINDOW_FREERATIO);
	namedWindow("��Ե����", WINDOW_FREERATIO);
	namedWindow("��Ե��ʴ", WINDOW_FREERATIO);
	//namedWindow("��������ͷ��Ե���", WINDOW_FREERATIO);

	imshow("ͼƬ", img);
	imshow("��˹ģ��", imgblur);
	imshow("������ؼ��", imgCanny); 
	imshow("��Ե����", imgDil);
	imshow("��Ե��ʴ", imgErode);

	while (true)
	{
		cap.read(imgs);
		Canny(imgs, imgCanny, 50, 150);
		imshow("��������ͷ��Ե���", imgCanny);
		waitKey(1);
	}


	waitKey(0);		//�ӳ���ʾ
}