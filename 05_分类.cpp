#include <opencv2/imgproc/imgproc.hpp> 
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector> 
#include <math.h>

using namespace cv;
using namespace std;

int main()
{
	Mat srcImage = imread("D://pp//yy.png");
	if (!srcImage.data)
	{
		printf("could not load image...\n");
		return -1;
	}
	imshow("srcImage", srcImage);
	Mat resultImag = srcImage.clone();
	//��ֵ�˲�
	medianBlur(srcImage, srcImage, 3);
	//ת����HSV��ɫ�ռ�
	Mat hsvImage;
	cvtColor(srcImage, hsvImage, CV_BGR2HSV);
	imshow("hsv", hsvImage);
	//��ɫ��ֵ������
	//����ߵ���ֵ
	Mat lowMat;
	Mat upperMat;
	//��ɫH�ķ�Χ��[0,10]  [160,180]
	inRange(hsvImage, Scalar(34, 30, 214), Scalar(34, 123, 255), lowMat);//��ɫ
	inRange(hsvImage, Scalar(34, 30, 214), Scalar(34, 123, 255), upperMat);
	imshow("lowMat", lowMat);
	imshow("upperMat", upperMat);
	//���ߵ���ֵ�ϲ�
	Mat redMat;
	addWeighted(lowMat, 1, upperMat, 1, 0, redMat);
	imshow("redMat", redMat);
	//��˹�˲�
	GaussianBlur(redMat, redMat, Size(9, 9), 2, 2);
	//����Բ�Ӳ�
	vector<Vec3f>  circles;
	HoughCircles(redMat, circles, CV_HOUGH_GRADIENT, 1, redMat.rows / 8, 100, 20, 0, 0);
  //HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//���û�м�⵽Բ
	if (circles.size() == 0)
		//return -1;
	for (int i = 0; i < circles.size(); i++)
	{
		//���Բ�ĵ�λ�ú�Բ�뾶�Ĵ�С
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(resultImag, center, radius, Scalar(0, 255, 0), 5);
	}
	imshow("resultImag", resultImag);
	waitKey(0);
	return 0;
}




//string path = "D://pp//yy.png";
	//Mat img = imread(path);	//���ļ�
	//Mat imggray, imgblur, imgCanny, imgDil;	//����ͼ�����


	//cvtColor(img, imggray, COLOR_BGR2GRAY);
	////��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�

	//GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	////����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)

	//Canny(imgblur, imgCanny, 30, 100);
	////��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)

	//namedWindow("ԭͼ", WINDOW_FREERATIO);
	////namedWindow("������ؼ��", WINDOW_FREERATIO);

	//imshow("ԭͼ", img);
	////imshow("������ؼ��", imgCanny);

	//Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	////�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��,ֻ��������)
	//dilate(imgCanny, imgDil, kernel);
	////��	��Ե����
	////imshow("��Ե����", imgDil);



	//cv2.kmeans(
	//	InputArray data,
	//	int K,
	//	InputOutputArray bestLabels,
	//	TermCriteria criteria,
	//	int attempts,
	//	int flags,
	//	OutputArray centers = noArray()
	//)



	//imshow("��Ч��ͼ��", img);


	//waitKey(0);		//�ӳ���ʾ
	// 
	// -----------------------------------------------------------------------------
//-----����Բ-----
	//std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ
	//	//����Բ
	//HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	////																		
	//for (size_t i = 0; i < circles.size(); i++)
	//{
	//	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	//	int radius = cvRound(circles[i][2]);

	//	//����Բ����  
	//	circle(img, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	//	int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
	//	int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
	//	int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
	//	std::cout << "Բ�İ뾶��" << radius << std::endl;
	//	std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
	//}