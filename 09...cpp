#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	//-----





	string path = "D://pp//yy.png";
	Mat img = imread(path);	//���ļ�
	Mat imggray, imgblur, imgCanny, imgDil;	//����ͼ�����


	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�
	imshow("�Ҷ�", imggray);
	imwrite("D://����//3636.png", imggray);


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
		
		
		
		//Point center(circles[i][0], circles[i][1]);
		//int radius = 200;

		circle(img, center, radius, Scalar(0, 200, 100), 2, 8, 0);

		for (int x = 0; x < circles[i][0]; x++)
		{
			for (int y = 0; y < circles[i][1]; y++)
			{
				int temp = ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
				if (temp < (radius * radius))
				{
					img.at<Vec3b>(Point(x, y))[0] = img.at<Vec3b>(Point(x, y))[0];
					img.at<Vec3b>(Point(x, y))[1] = img.at<Vec3b>(Point(x, y))[1];
					img.at<Vec3b>(Point(x, y))[2] = img.at<Vec3b>(Point(x, y))[2];
				}
				else
				{
					img.at<Vec3b>(Point(x, y))[0] = img.at<Vec3b>(Point(x, y))[0];
					img.at<Vec3b>(Point(x, y))[1] = img.at<Vec3b>(Point(x, y))[1];
					img.at<Vec3b>(Point(x, y))[2] = img.at<Vec3b>(Point(x, y))[2];
				}
			}
		}

		imshow("image1", img);
		imshow("image2", img);

		while (uchar(waitKey() != 'q')) {}
		return 0;
	}






	

}





//#include<opencv2/imgcodecs.hpp>
//#include<opencv2/highgui.hpp>
//#include<opencv2/imgproc.hpp>
//#include<iostream>
//#include <thread>
//#include <string>
//#include <vector>
//#include <opencv2/imgproc/imgproc.hpp>  
//
//using namespace cv;
//using namespace std;
//
//void main() {
//
//	string path = "D://pp//yy.png";
//	Mat img = imread(path);	//���ļ�
//	Mat imggray, imgblur, imgCanny, imgDil;	//����ͼ�����
//
//
//	cvtColor(img, imggray, COLOR_BGR2GRAY);
//	//��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�
//	imshow("�Ҷ�", imggray);
//
//
//	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
//	//����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)
//
//	Canny(imgblur, imgCanny, 30, 100);
//	//��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)
//
//
//	namedWindow("ԭͼ", WINDOW_FREERATIO);
//	namedWindow("������ؼ��", WINDOW_FREERATIO);
//
//	imshow("ԭͼ", img);
//	imshow("������ؼ��", imgCanny);
//
//	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
//	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��,ֻ��������)
//	dilate(imgCanny, imgDil, kernel);
//	//��	��Ե����
//	imshow("��Ե����", imgDil);
//
//	std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ
//		//����Բ
//	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
//	//																		
//	for (size_t i = 0; i < circles.size(); i++)
//	{
//		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//		int radius = cvRound(circles[i][2]);
//
//		//����Բ����  
//		circle(img, center, radius, Scalar(155, 50, 255), 3, 8, 0);
//		int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
//		int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
//		int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
//		std::cout << "Բ�İ뾶��" << radius << std::endl;
//		std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
//	}
//
//	imshow("��Ч��ͼ��", img);
//
//	waitKey(0);		//�ӳ���ʾ
//}
