#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/我的文件/图片/文字/QQphoto (3).jpeg";

	Mat img = imread(path);
	Mat imgCrop;


	Rect roi(120, 260, 400, 320);
	//     Rect -> 定义矩形数据类型  名称(左上x坐标,左上y坐标,矩形宽，矩形高)

	imgCrop = img(roi);
	//		    ↑将矩形放入图像中


	cout << "原图像大小:  " << img.size() << endl;
	cout << "裁剪后图像大小:" << imgCrop.size() << endl;


	imshow("原图像", img);
	imshow("剪裁图像", imgCrop);



	waitKey(0);		//延迟显示
}