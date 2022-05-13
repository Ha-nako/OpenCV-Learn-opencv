#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>//可使用联级文件
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/A_file/1.jpg";
	Mat img = imread(path);

	CascadeClassifier faceCascade;//级联分类器(面对层叠)
	faceCascade.load("D:/RJDZ/opencv_cpp_course_resources/resources/haarcascade_frontalface_default.xml");
	//↑(面对级联点负载)【脸部文件训练路径(英文)】

	if (faceCascade.empty()){cout << "XML file not loadde... " << endl;}
	//↑检查文件是否成功加载

	vector<Rect>faces;//创建矩形向量

	faceCascade.detectMultiScale(img, faces, 1.1, 10);//【人脸检测 : img(脸) -> faces】
	// 1.1:前后两次搜索中搜索窗口比例系数, 10:构成检测目标的相邻矩形的最小个数

	for (int i = 0; i < faces.size(); i++)//遍历每张人脸
	{
		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255),20);//画矩形
						//面部左上角	右下角
	}

	namedWindow("Image", WINDOW_FREERATIO);
	imshow("Image", img);

	waitKey(0);
}