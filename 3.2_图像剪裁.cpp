#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/�ҵ��ļ�/ͼƬ/����/QQphoto (3).jpeg";

	Mat img = imread(path);
	Mat imgCrop;


	Rect roi(120, 260, 400, 320);
	//     Rect -> ���������������  ����(����x����,����y����,���ο����θ�)

	imgCrop = img(roi);
	//		    �������η���ͼ����


	cout << "ԭͼ���С:  " << img.size() << endl;
	cout << "�ü���ͼ���С:" << imgCrop.size() << endl;


	imshow("ԭͼ��", img);
	imshow("����ͼ��", imgCrop);



	waitKey(0);		//�ӳ���ʾ
}