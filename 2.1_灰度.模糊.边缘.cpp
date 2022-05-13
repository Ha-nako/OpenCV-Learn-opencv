#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {
				//������ͼƬ·��
	//string path = "D:/�ҵ��ļ�/ͼƬ/�����ز�/��Ǿ�ͷ����.jpg";	
	string path = "D://pp//yy.png";
	Mat img = imread(path);	//���ļ�
	Mat imggray,imgblur,imgCanny;	//����ͼ�����


	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)
	
	Canny(imgblur, imgCanny, 30, 100);
	//��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)

	
	namedWindow("ͼƬ", WINDOW_FREERATIO);
	namedWindow("�Ҷ�ͼƬ", WINDOW_FREERATIO);
	namedWindow("��˹ģ��", WINDOW_FREERATIO);
	namedWindow("������ؼ��", WINDOW_FREERATIO);

	imshow("ͼƬ", img);			//���
	imshow("�Ҷ�ͼƬ", imggray);
	imshow("��˹ģ��", imgblur);
	imshow("������ؼ��", imgCanny);
	

	waitKey(0);		//�ӳ���ʾ
}