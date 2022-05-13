#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>//��ʹ�������ļ�
#include<iostream>
#include<string>
#include<time.h>
#include <windows.h>

using namespace cv;
using namespace std;

void main() {

	int mm;
	cout << "����1����ʼ����沿��";
	cin >> mm;

	if (mm == 1)
	{
		cout << endl << endl;
		cout << "���ڵ�������ͷ..." << endl << "�����ɹ�..." << endl << endl;
		goto MM;
	}
	else
	{
		cout << "������� !" << endl;
		system("pause");
		exit(0);
	}

MM:
	VideoCapture cap(0);

	Mat img;

	CascadeClassifier faceCascade;//����������(��Բ��)
	faceCascade.load("D:/RJDZ/opencv_cpp_course_resources/resources/haarcascade_frontalface_default.xml");
	//��(��Լ����㸺��)�������ļ�ѵ��·��(Ӣ��)�����������

	if (faceCascade.empty()) { cout << "XML file not loadde... " << endl; }
	//������ļ��Ƿ�ɹ�����

	vector<Rect>faces;//������������

	namedWindow("Image", WINDOW_FREERATIO);



	while (1)
	{
		cap >> img;
		faceCascade.detectMultiScale(img, faces, 1.1, 10);

		//------ʱ��-------------
		SYSTEMTIME sys;
		GetLocalTime(&sys);
		int A = sys.wHour;
		int B = sys.wMinute;
		int C = sys.wSecond;
		//-----------------------


		for (int i = 0; i < faces.size(); i++)//����ÿ������
		{
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 5);//������
							//�沿���Ͻ�	���½�
			putText(img, "Time:"+to_string(A) + ":" + to_string(B) + ":" + to_string(C), Point(faces[i].x, (faces[i].y) - 5), FONT_HERSHEY_PLAIN, 1, Scalar(50, 205, 50), 1);
		}//"Handsome Boy" + to_string(i)

		//for (int i = 0; i < faces.size(); i++)
		//{
		//	Mat imgCrop = img(faces[i]);//����
		//	imshow(to_string(i), imgCrop);
		//	imwrite("D:/����/ " + to_string(i) + ".png", imgCrop);
		//	rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);//����
		//}


		//flip(img, img, 1);//����ͷͼ��y�ᷭת
		//namedWindow("Image", WINDOW_FREERATIO);
		imshow("Image", img);

		waitKey(1);
		
		//printf("%02d:%02d:%02d.\n", , , );
		
	}
}