#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>//��ʹ�������ļ�
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/A_file/1.jpg";
	Mat img = imread(path);

	CascadeClassifier faceCascade;//����������(��Բ��)
	faceCascade.load("D:/RJDZ/opencv_cpp_course_resources/resources/haarcascade_frontalface_default.xml");
	//��(��Լ����㸺��)�������ļ�ѵ��·��(Ӣ��)��

	if (faceCascade.empty()){cout << "XML file not loadde... " << endl;}
	//������ļ��Ƿ�ɹ�����

	vector<Rect>faces;//������������

	faceCascade.detectMultiScale(img, faces, 1.1, 10);//��������� : img(��) -> faces��
	// 1.1:ǰ�������������������ڱ���ϵ��, 10:���ɼ��Ŀ������ھ��ε���С����

	for (int i = 0; i < faces.size(); i++)//����ÿ������
	{
		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255),20);//������
						//�沿���Ͻ�	���½�
	}

	namedWindow("Image", WINDOW_FREERATIO);
	imshow("Image", img);

	waitKey(0);
}