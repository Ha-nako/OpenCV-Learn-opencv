#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/�ҵ��ļ�/ͼƬ/����/QQphoto (3).jpeg";	

	Mat img = imread(path);
	Mat imgResize, imgresize;
	
	resize(img, imgResize, Size(500, 500));
	//					��Size������ĺ��С

	resize(img, imgresize, Size(), 0.5, 0.5);
	//					��x,y ��ֵ 0.5,0.5	��	

	

	cout << "ԭͼ���С:  " << img.size() << endl;
	cout << "��ֵ������ͼ���С:  " << imgResize.size() << endl;
	cout << "����������ͼ���С:  " << imgresize.size() << endl;					

	imshow("ԭͼ��", img);	
	imshow("��ֵ������ͼ��", imgResize);
	imshow("����������ͼ��", imgresize);


	
	waitKey(0);		//�ӳ���ʾ
}