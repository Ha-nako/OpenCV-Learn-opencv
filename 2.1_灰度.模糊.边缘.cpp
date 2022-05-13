#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {
				//↓访问图片路径
	//string path = "D:/我的文件/图片/所用素材/广角镜头拍摄.jpg";	
	string path = "D://pp//yy.png";
	Mat img = imread(path);	//读文件
	Mat imggray,imgblur,imgCanny;	//定义图像变量


	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//↑转化灰度图像:1转化为2,模式为3（灰度）

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//↑高斯模糊:size(定义内核大小7*7),输出屏幕上的位置偏移->5,0(x,y)
	
	Canny(imgblur, imgCanny, 30, 100);
	//↑坎尼边沿检测	  两个检测阈值(可更改,值越小边缘越多)

	
	namedWindow("图片", WINDOW_FREERATIO);
	namedWindow("灰度图片", WINDOW_FREERATIO);
	namedWindow("高斯模糊", WINDOW_FREERATIO);
	namedWindow("坎尼边沿检测", WINDOW_FREERATIO);

	imshow("图片", img);			//输出
	imshow("灰度图片", imggray);
	imshow("高斯模糊", imgblur);
	imshow("坎尼边沿检测", imgCanny);
	

	waitKey(0);		//延迟显示
}