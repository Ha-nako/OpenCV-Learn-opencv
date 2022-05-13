#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;
using namespace std;

Mat getHistograph(const Mat grayImage);
void GetGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev);
int Lei = 0;

int main() {

	string path = "D://pp//yy.bmp";
	Mat img = imread(path);	//读文件
	//imshow("原图", img);

	double a, b;

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
	//↑定义可使用膨胀(扩充/侵蚀)的内核  (数越小,扩充越多,只能用奇数)
	dilate(imgCanny, imgDil, kernel);
	//↑	边缘扩充
	//imshow("边缘扩充", imgDil);

	std::vector<Vec3f> circles;//存储每个圆的位置信息
		//霍夫圆
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//	
	Mat imgCrop, hist;
	for (size_t i = 0; i < circles.size(); i++)
	{
		cout << endl << "第 " << i + 1 << " 个圆：" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//绘制圆轮廓  
		circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);
		//int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
		//int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
		//int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
		std::cout << "圆的半径是" << radius << std::endl;
		std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;


		//Mat image = imread(IMG_PATH);
		Mat dst = Mat::zeros(img.size(), img.type());
		Mat mask = Mat::zeros(img.size(), CV_8U);


		Rect roi(circles[i][0] - radius, circles[i][1] - radius, 2 * radius + 3, 2 * radius + 1);
		//     Rect -> 定义矩形数据类型  名称(左上x坐标,左上y坐标,矩形宽，矩形高)

		imgCrop = imggray(roi);
		//		    ↑将矩形放入图像中

		hist = getHistograph(imgCrop);

		//灰度均值、方差
		GetGrayAvgStdDev(imgCrop,a,b);
		int bb = (int)b;
		putText(img,to_string(bb), Point(circles[i][0] - 10, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 0, 0), 0.5);
		
		//文字
		//if (Lei == 1)
		//{
		//	putText(img, "1", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑
		//}
		//if (Lei == 2)
		//{
		//	putText(img, "2", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑
		//}
		//if (Lei == 3)
		//{
		//	putText(img, "3", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑
		//}
		Lei = 0;

		//imshow("hist"+i, hist);
		//imshow("剪裁图像+"+i, imgCrop);

	}
	imshow("【效果图】", img);
	waitKey(0);		//延迟显示
	return 0;
}

Mat getHistograph(const Mat grayImage)
{
	int sum1 = 0;
	int sum2 = 0;
	int sum3 = 0;

	//定义求直方图的通道数目，从0开始索引
	int channels[] = { 0 };
	//定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数
	//如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目
	const int histSize[] = { 256 };
	//每一维bin的变化范围
	float range[] = { 0,256 };

	//所有bin的变化范围，个数跟channels应该跟channels一致
	const float* ranges[] = { range };

	//定义直方图，这里求的是直方图数据
	Mat hist;
	//opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数
	calcHist(&grayImage, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);//cv中是cvCalcHist

	//找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
	double maxValue = 0;
	//找矩阵中最大最小值及对应索引的函数
	minMaxLoc(hist, 0, &maxValue, 0, 0);
	//最大值取整
	int rows = cvRound(maxValue);
	//定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
	//因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
	Mat histImage = Mat::zeros(rows, 256, CV_8UC1);

	//直方图图像表示
	for (int i = 0; i < 256; i++)
	{
		//取每个bin的数目
		int temp = (int)(hist.at<float>(i, 0));
		//如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
		//如果图像上有该灰度值，则将该列对应个数的像素设为白色
		if (temp)
		{
			//由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
			histImage.col(i).rowRange(Range(rows - temp, rows)) = 255;
		}
		//if (150 < i < 200)
		//{
		//	sum2 = sum2 + temp;
		//}
		if (i > 155)
		{
			sum3 = sum3 + temp;
		}
		if (i > 200)
		{
			sum1 = sum1 + temp;
		}
	}
	//判断通道内像素点个数
	if (sum1 > 100)
	{
		Lei = 1;
	}
	else if (sum3 == 0)
	{
		Lei = 3;
	}
	else
	{
		Lei = 2;
	}

	//由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
	Mat resizeImage;
	resize(histImage, resizeImage, Size(256, 256));
	return resizeImage;
}

//计算灰度均值及方差
void GetGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev)
{
	cv::Mat img;
	if (src.channels() == 3)
		cv::cvtColor(src, img, CV_BGR2GRAY);
	else
		img = src;
	cv::mean(src);
	cv::Mat mean;
	cv::Mat stdDev;
	cv::meanStdDev(img, mean, stdDev);

	avg = mean.ptr<double>(0)[0];
	stddev = stdDev.ptr<double>(0)[0];
}




