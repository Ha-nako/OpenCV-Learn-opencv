#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/�ҵ��ļ�/ͼƬ/�����ز�/�Զ�����/3.jpg";	
	Mat img = imread(path);	
	Mat imgresize;
	Mat matrix, imgWarp;
	resize(img, imgresize, Size(), 0.6, 0.5);
	

	float w = 350, h = 230;
	Point2f	src[4] = { {820,889},{960,1040},{674,943},{857,1127} };//Դͼ���������
//��������	��Դ

	Point2f	dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };//��ͼ���������
//						����		����	����		����


	matrix = getPerspectiveTransform(src, dst);//��1ͶӰ��2
		//͸�ӱ任����
	warpPerspective(img, imgWarp, matrix, Point(w, h));
	//͸�ӱ任ʵ��
	
	for (int i = 0; i < 4; i++)
	{
		circle(img, src[i], 10, Scalar(0, 0, 255), FILLED);
	}


	namedWindow("ԭͼ", WINDOW_FREERATIO);

	imshow("͸��", imgWarp);
	imshow("ԭͼ", img);

	waitKey(0);	
}