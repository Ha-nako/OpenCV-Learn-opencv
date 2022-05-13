#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;

void SendMessageOne()
{
	//��������ͷ
	VideoCapture capture;
	capture.open(0);
	Mat edges;  //����ת���ĻҶ�ͼ
	if (!capture.isOpened())
		namedWindow("��Ч��ͼ��", CV_WINDOW_NORMAL);

	while (1)
	{
		int Y = 0, J = 0;
		Mat frame;
		capture >> frame;
		cvtColor(frame, edges, CV_BGR2GRAY);
		//��˹�˲�
		GaussianBlur(edges, edges, Size(7, 7), 2, 2);
		std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ
		//����Բ
		HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 1.5, 10, 100, 100, 0, 100);
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);

			//����Բ����  
			circle(frame, center, radius, Scalar(155, 50, 255), 3, 8, 0);
			int R = frame.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
			int G = frame.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
			int B = frame.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
			std::cout << "Բ�İ뾶��" << radius << std::endl;
			std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
		}

		imshow("��Ч��ͼ��", frame);
		waitKey(30);
	}
}

int main()
{
	std::thread* a = new std::thread(SendMessageOne);
	a->join();
	return 0;
}