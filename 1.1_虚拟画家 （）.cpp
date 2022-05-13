#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

vector<vector<int>>myColor{ {100,43,46,124,255,255},//蓝色	[类似二维数组]
							{0,0,0,180,255,46} };//	黑色	
	//↑矢量(HSV)			    	↑将HSV数值粘贴(颜色选择器)可加入新颜色

vector<Scalar>myColorValues{ {255,191,0}//蓝色
								,{0,0,0} };// 黑色
//	↑标量(BGR)

VideoCapture cap(0);

Mat img;
vector<vector<int>>newPoints;

Point getContours(Mat imgDil) {		//轮廓（输入mask）

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	Point myPoint(0, 0);

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		cout << area << endl;
		string objectType;

		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			cout << conPoly[i].size() << endl;
			boundRect[i] = boundingRect(conPoly[i]);
			myPoint.x = boundRect[i].x + boundRect[i].width / 2;
			myPoint.y = boundRect[i].y;

			//drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);	
			//rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
		}
	}
	return myPoint;
}


vector<vector<int>> findColor(Mat img)

{
	Mat imgHSV, mask;

	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	for (int i = 0; i < myColor.size(); i++)//将每种颜色变成遮罩
	{
		Scalar lower(myColor[i][0], myColor[i][1], myColor[i][2]);
		Scalar upper(myColor[i][3], myColor[i][4], myColor[i][5]);
		inRange(imgHSV, lower, upper, mask);//提取颜色范围并输出遮罩后图像★
		//imshow(to_string(i), mask);// to_string()	将数值转换成字符串
		getContours(mask);
		Point myPoint = getContours(mask);
		if (myPoint.x != 0)
		{
			newPoints.push_back({ myPoint.x,myPoint.y,i });
		}	//			↑将一个新的元素加到vector的最后面(类似py的append)
		
	}
	return newPoints;
}

void drawOnCanvas(vector<vector<int>> newPoints, vector<Scalar> myColorValues)
{

	for (int i = 0; i < newPoints.size(); i++)
	{
		circle(img, Point(newPoints[i][0], newPoints[i][1]), 10, myColorValues[newPoints[i][2]], FILLED);
	}
}//画圆


void main() {

	
	while (true)
	{
		cap >> img;

		newPoints=findColor(img);
		drawOnCanvas(newPoints,myColorValues);

		imshow("video", img);
		waitKey(1);	
	}
}