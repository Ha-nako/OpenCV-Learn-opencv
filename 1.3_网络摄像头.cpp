#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	VideoCapture cap(0);//	�����ID,ֻ��һ����0
	Mat img;

	while (true)
	{
		cap.read(img);	//��ȡΪһ֡֡��ͼ��	�� cap >> img ;

		flip(img, img, 1);//����ʵ��ͼ��ת�����������룬�����������1Ϊy�ᷴת��0Ϊx�ᷴת������Ϊx,y��ת����
		
		imshow("video", img);
		waitKey(1);		//���1ms,��ֹ�����ӳٿ���
	}
}