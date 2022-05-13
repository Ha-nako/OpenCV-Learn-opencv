#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>//可使用联级文件
#include<iostream>
#include<string>
#include<time.h>
#include <windows.h>

using namespace cv;
using namespace std;

void main() {

	int mm;
	cout << "输入1，开始检测面部：";
	cin >> mm;

	if (mm == 1)
	{
		cout << endl << endl;
		cout << "正在调用摄像头..." << endl << "启动成功..." << endl << endl;
		goto MM;
	}
	else
	{
		cout << "程序结束 !" << endl;
		system("pause");
		exit(0);
	}

MM:
	VideoCapture cap(0);

	Mat img;

	CascadeClassifier faceCascade;//级联分类器(面对层叠)
	faceCascade.load("D:/RJDZ/opencv_cpp_course_resources/resources/haarcascade_frontalface_default.xml");
	//↑(面对级联点负载)【脸部文件训练路径(英文)】人脸检测器

	if (faceCascade.empty()) { cout << "XML file not loadde... " << endl; }
	//↑检查文件是否成功加载

	vector<Rect>faces;//创建矩形向量

	namedWindow("Image", WINDOW_FREERATIO);



	while (1)
	{
		cap >> img;
		faceCascade.detectMultiScale(img, faces, 1.1, 10);

		//------时间-------------
		SYSTEMTIME sys;
		GetLocalTime(&sys);
		int A = sys.wHour;
		int B = sys.wMinute;
		int C = sys.wSecond;
		//-----------------------


		for (int i = 0; i < faces.size(); i++)//遍历每张人脸
		{
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 5);//画矩形
							//面部左上角	右下角
			putText(img, "Time:"+to_string(A) + ":" + to_string(B) + ":" + to_string(C), Point(faces[i].x, (faces[i].y) - 5), FONT_HERSHEY_PLAIN, 1, Scalar(50, 205, 50), 1);
		}//"Handsome Boy" + to_string(i)

		//for (int i = 0; i < faces.size(); i++)
		//{
		//	Mat imgCrop = img(faces[i]);//裁切
		//	imshow(to_string(i), imgCrop);
		//	imwrite("D:/桌面/ " + to_string(i) + ".png", imgCrop);
		//	rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);//矩形
		//}


		//flip(img, img, 1);//摄像头图像y轴翻转
		//namedWindow("Image", WINDOW_FREERATIO);
		imshow("Image", img);

		waitKey(1);
		
		//printf("%02d:%02d:%02d.\n", , , );
		
	}
}