#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  
#include<math.h>
#include<opencv2/opencv.hpp>


using namespace cv;
using namespace std;


int main() {

	//ͼƬ��ȡ·��
	string path = "D://pp//yy.bmp";
	Mat img = imread(path);
	//imshow("ԭͼ", img);

	if (img.empty())
	{
		cerr << "δ�ҵ��ļ�������" << endl;
		return -1;
	}


	Mat imggray, imgblur, imgCanny, imgDil;	//����ͼ�����

	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�
	//imshow("�Ҷ�", imggray);

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)

	Canny(imgblur, imgCanny, 30, 100);
	//��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)

	//imshow("ԭͼ", img);
	//imshow("������ؼ��", imgCanny);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��)
	dilate(imgCanny, imgDil, kernel);
	//��	��Ե����
	//imshow("��Ե����", imgDil);

	std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ

		//����Բ
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);



	//+++++++++++++++++++++++++ð������1+++++++++++++++++++++++++++++++
	int t, tt;
	for (int i = 0; i < 96; i++)
	{
		for (int j = 0; j < 96 - i - 1; j++)
		{
			if (circles[j][0] > circles[j + 1][0])
			{
				t = circles[j][0];
				tt = circles[j][1];

				circles[j][0] = circles[j + 1][0];
				circles[j][1] = circles[j + 1][1];

				circles[j + 1][0] = t;
				circles[j + 1][1] = tt;

			}
		}
	}

	//-------------------
	int k = 0, s = 0;

	for (k = 0; k < 12; k++)
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = 1; j < 8 - i; j++)
			{

				//if (circles[j ][1] > circles[j + 1 ][1])
				if (circles[k * 8 + i][1] > circles[k * 8 + i + j][1])
				{
					t = circles[k * 8 + i][1];
					tt = circles[k * 8 + i][0];

					circles[k * 8 + i][0] = circles[k * 8 + i + j][0];
					circles[k * 8 + i][1] = circles[k * 8 + i + j][1];

					circles[k * 8 + i + j][1] = t;
					circles[k * 8 + i + j][0] = tt;
				}
				//cout << endl << "j " << j  << endl;
			}

		}
	}
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	//��������Բ��
	for (size_t i = 0; i < circles.size(); i++)
	{
		//cout << endl << "�� " << i + 1 << " ��Բ��" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//����Բ����  
		//circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);

		//std::cout << "Բ�İ뾶��" << radius << std::endl;
		//std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
		//cout << "----------------------------------------" << endl;


		//�����Բ��
		putText(img, to_string(i), Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0, 0, 255), 1.2);
//					   ���������		 ��x����λ��		  �� y����λ��									����ɫ			 �����ֺ��				
		



	}
	imshow("��Ч��ͼ��", img);
	waitKey(0);

	return 0;
}

