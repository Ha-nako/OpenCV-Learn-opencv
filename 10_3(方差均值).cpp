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

Mat getHistograph(const Mat grayImage);
void GetGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev);
int Lei = 0;

int main() {

	string path = "D://pp//yy.bmp";
	Mat img = imread(path);	//���ļ�
	//imshow("ԭͼ", img);

	double a, b;

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
	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��,ֻ��������)
	dilate(imgCanny, imgDil, kernel);
	//��	��Ե����
	//imshow("��Ե����", imgDil);

	std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ
		//����Բ
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//	
	Mat imgCrop, hist;
	for (size_t i = 0; i < circles.size(); i++)
	{
		cout << endl << "�� " << i + 1 << " ��Բ��" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//����Բ����  
		circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);
		//int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
		//int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
		//int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
		std::cout << "Բ�İ뾶��" << radius << std::endl;
		std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;


		//Mat image = imread(IMG_PATH);
		Mat dst = Mat::zeros(img.size(), img.type());
		Mat mask = Mat::zeros(img.size(), CV_8U);


		Rect roi(circles[i][0] - radius, circles[i][1] - radius, 2 * radius + 3, 2 * radius + 1);
		//     Rect -> ���������������  ����(����x����,����y����,���ο����θ�)

		imgCrop = imggray(roi);
		//		    �������η���ͼ����

		hist = getHistograph(imgCrop);

		//�ҶȾ�ֵ������
		GetGrayAvgStdDev(imgCrop,a,b);
		int bb = (int)b;
		putText(img,to_string(bb), Point(circles[i][0] - 10, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 0, 0), 0.5);
		
		//����
		//if (Lei == 1)
		//{
		//	putText(img, "1", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	���֣�	  ������					       �����壨�����		������		      ��ȡ�
		//}
		//if (Lei == 2)
		//{
		//	putText(img, "2", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	���֣�	  ������					       �����壨�����		������		      ��ȡ�
		//}
		//if (Lei == 3)
		//{
		//	putText(img, "3", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	���֣�	  ������					       �����壨�����		������		      ��ȡ�
		//}
		Lei = 0;

		//imshow("hist"+i, hist);
		//imshow("����ͼ��+"+i, imgCrop);

	}
	imshow("��Ч��ͼ��", img);
	waitKey(0);		//�ӳ���ʾ
	return 0;
}

Mat getHistograph(const Mat grayImage)
{
	int sum1 = 0;
	int sum2 = 0;
	int sum3 = 0;

	//������ֱ��ͼ��ͨ����Ŀ����0��ʼ����
	int channels[] = { 0 };
	//����ֱ��ͼ����ÿһά�ϵĴ�С������Ҷ�ͼֱ��ͼ�ĺ�������ͼ��ĻҶ�ֵ����һά��bin�ĸ���
	//���ֱ��ͼͼ�������bin����Ϊx��������bin����Ϊy����channels[]={1,2}��ֱ��ͼӦ��Ϊ��ά�ģ�Z����ÿ��bin��ͳ�Ƶ���Ŀ
	const int histSize[] = { 256 };
	//ÿһάbin�ı仯��Χ
	float range[] = { 0,256 };

	//����bin�ı仯��Χ��������channelsӦ�ø�channelsһ��
	const float* ranges[] = { range };

	//����ֱ��ͼ�����������ֱ��ͼ����
	Mat hist;
	//opencv�м���ֱ��ͼ�ĺ�����hist��СΪ256*1��ÿ�д洢��ͳ�Ƶĸ��ж�Ӧ�ĻҶ�ֵ�ĸ���
	calcHist(&grayImage, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);//cv����cvCalcHist

	//�ҳ�ֱ��ͼͳ�Ƶĸ��������ֵ��������Ϊֱ��ͼ������ĸ�
	double maxValue = 0;
	//�Ҿ����������Сֵ����Ӧ�����ĺ���
	minMaxLoc(hist, 0, &maxValue, 0, 0);
	//���ֵȡ��
	int rows = cvRound(maxValue);
	//����ֱ��ͼͼ��ֱ��ͼ������ĸ���Ϊ����������Ϊ256(�Ҷ�ֵ�ĸ���)
	//��Ϊ��ֱ��ͼ��ͼ�������Ժڰ���ɫΪ���֣���ɫΪֱ��ͼ��ͼ��
	Mat histImage = Mat::zeros(rows, 256, CV_8UC1);

	//ֱ��ͼͼ���ʾ
	for (int i = 0; i < 256; i++)
	{
		//ȡÿ��bin����Ŀ
		int temp = (int)(hist.at<float>(i, 0));
		//���bin��ĿΪ0����˵��ͼ����û�иûҶ�ֵ��������Ϊ��ɫ
		//���ͼ�����иûҶ�ֵ���򽫸��ж�Ӧ������������Ϊ��ɫ
		if (temp)
		{
			//����ͼ�������������Ͻ�Ϊԭ�㣬����Ҫ���б任��ʹֱ��ͼͼ�������½�Ϊ����ԭ��
			histImage.col(i).rowRange(Range(rows - temp, rows)) = 255;
		}
		//if (150 < i < 200)
		//{
		//	sum2 = sum2 + temp;
		//}
		if (i > 155)
		{
			sum3 = sum3 + temp;
		}
		if (i > 200)
		{
			sum1 = sum1 + temp;
		}
	}
	//�ж�ͨ�������ص����
	if (sum1 > 100)
	{
		Lei = 1;
	}
	else if (sum3 == 0)
	{
		Lei = 3;
	}
	else
	{
		Lei = 2;
	}

	//����ֱ��ͼͼ���и߿��ܸܺߣ���˽���ͼ�����Ҫ���ж�Ӧ��������ʹֱ��ͼͼ���ֱ��
	Mat resizeImage;
	resize(histImage, resizeImage, Size(256, 256));
	return resizeImage;
}

//����ҶȾ�ֵ������
void GetGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev)
{
	cv::Mat img;
	if (src.channels() == 3)
		cv::cvtColor(src, img, CV_BGR2GRAY);
	else
		img = src;
	cv::mean(src);
	cv::Mat mean;
	cv::Mat stdDev;
	cv::meanStdDev(img, mean, stdDev);

	avg = mean.ptr<double>(0)[0];
	stddev = stdDev.ptr<double>(0)[0];
}




