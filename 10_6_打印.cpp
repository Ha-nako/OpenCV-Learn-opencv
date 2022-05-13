#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  
#include<math.h>
#include<opencv2/opencv.hpp>


using namespace cv;
using namespace std;


int main() {

	//图片读取路径
	string path = "D://pp//yy.bmp";
	Mat img = imread(path);
	//imshow("原图", img);

	if (img.empty())
	{
		cerr << "未找到文件！！！" << endl;
		return -1;
	}


	Mat imggray, imgblur, imgCanny, imgDil;	//定义图像变量

	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//↑转化灰度图像:1转化为2,模式为3（灰度）
	//imshow("灰度", imggray);

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//↑高斯模糊:size(定义内核大小7*7),输出屏幕上的位置偏移->5,0(x,y)

	Canny(imgblur, imgCanny, 30, 100);
	//↑坎尼边沿检测	  两个检测阈值(可更改,值越小边缘越多)

	//imshow("原图", img);
	//imshow("坎尼边沿检测", imgCanny);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	//↑定义可使用膨胀(扩充/侵蚀)的内核  (数越小,扩充越多)
	dilate(imgCanny, imgDil, kernel);
	//↑	边缘扩充
	//imshow("边缘扩充", imgDil);

	std::vector<Vec3f> circles;//存储每个圆的位置信息

		//霍夫圆
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);



	//+++++++++++++++++++++++++冒泡排序1+++++++++++++++++++++++++++++++
	int t, tt;
	for (int i = 0; i < 96; i++)
	{
		for (int j = 0; j < 96 - i - 1; j++)
		{
			if (circles[j][0] > circles[j + 1][0])
			{
				t = circles[j][0];
				tt = circles[j][1];

				circles[j][0] = circles[j + 1][0];
				circles[j][1] = circles[j + 1][1];

				circles[j + 1][0] = t;
				circles[j + 1][1] = tt;

			}
		}
	}

	//-------------------
	int k = 0, s = 0;

	for (k = 0; k < 12; k++)
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = 1; j < 8 - i; j++)
			{

				//if (circles[j ][1] > circles[j + 1 ][1])
				if (circles[k * 8 + i][1] > circles[k * 8 + i + j][1])
				{
					t = circles[k * 8 + i][1];
					tt = circles[k * 8 + i][0];

					circles[k * 8 + i][0] = circles[k * 8 + i + j][0];
					circles[k * 8 + i][1] = circles[k * 8 + i + j][1];

					circles[k * 8 + i + j][1] = t;
					circles[k * 8 + i + j][0] = tt;
				}
				//cout << endl << "j " << j  << endl;
			}

		}
	}
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	//遍历所有圆：
	for (size_t i = 0; i < circles.size(); i++)
	{
		//cout << endl << "第 " << i + 1 << " 个圆：" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//绘制圆轮廓  
		//circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);

		//std::cout << "圆的半径是" << radius << std::endl;
		//std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;
		//cout << "----------------------------------------" << endl;


		//输出到圆上
		putText(img, to_string(i), Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0, 0, 255), 1.2);
//					   ↑输出变量		 ↑x坐标位置		  ↑ y坐标位置									↑颜色			 ↑文字厚度				
		



	}
	imshow("【效果图】", img);
	waitKey(0);

	return 0;
}

