#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/�ҵ��ļ�/ͼƬ/�����ز�/��Ǿ�ͷ����.jpg";	
	Mat img = imread(path);	
	namedWindow("ͼƬ", WINDOW_FREERATIO);
	imshow("ͼƬ", img);	


	waitKey(0);		
}