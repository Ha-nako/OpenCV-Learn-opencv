#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;
using namespace std;

void main() {

	string path = "D://pp//yy.png";
	Mat img = imread(path);	//���ļ�
	Mat imggray, imgblur, imgCanny, imgDil;	//����ͼ�����

	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�
	imshow("�Ҷ�", imggray);

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)

	Canny(imgblur, imgCanny, 30, 100);
	//��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)


	namedWindow("ԭͼ", WINDOW_FREERATIO);
	namedWindow("������ؼ��", WINDOW_FREERATIO);

	imshow("ԭͼ", img);
	imshow("������ؼ��", imgCanny);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��,ֻ��������)
	dilate(imgCanny, imgDil, kernel);
	//��	��Ե����
	imshow("��Ե����", imgDil);

	std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ
		//����Բ
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//																		
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		//����Բ����  
		circle(img, center, radius, Scalar(155, 50, 255), 3, 8, 0);
		int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
		int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
		int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
		std::cout << "Բ�İ뾶��" << radius << std::endl;
		std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
	}

	imshow("��Ч��ͼ��", img);


	waitKey(0);		//�ӳ���ʾ
}