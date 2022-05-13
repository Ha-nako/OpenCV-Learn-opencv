#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/我的文件/图片/所用素材/自动对齐/3.jpg";	
	Mat img = imread(path);	
	Mat imgresize;
	Mat matrix, imgWarp;
	resize(img, imgresize, Size(), 0.6, 0.5);
	

	float w = 350, h = 230;
	Point2f	src[4] = { {820,889},{960,1040},{674,943},{857,1127} };//源图像矩阵坐标
//浮点数↑	↑源

	Point2f	dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };//后图像矩阵坐标
//						左上		右上	左下		右下


	matrix = getPerspectiveTransform(src, dst);//将1投影到2
		//透视变换矩阵
	warpPerspective(img, imgWarp, matrix, Point(w, h));
	//透视变换实现
	
	for (int i = 0; i < 4; i++)
	{
		circle(img, src[i], 10, Scalar(0, 0, 255), FILLED);
	}


	namedWindow("原图", WINDOW_FREERATIO);

	imshow("透视", imgWarp);
	imshow("原图", img);

	waitKey(0);	
}