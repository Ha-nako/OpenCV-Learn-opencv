#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/我的文件/图片/文字/QQphoto (3).jpeg";	

	Mat img = imread(path);
	Mat imgResize, imgresize;
	
	resize(img, imgResize, Size(500, 500));
	//					↑Size内填更改后大小

	resize(img, imgresize, Size(), 0.5, 0.5);
	//					↑x,y 比值 0.5,0.5	★	

	

	cout << "原图像大小:  " << img.size() << endl;
	cout << "数值调整后图像大小:  " << imgResize.size() << endl;
	cout << "比例调整后图像大小:  " << imgresize.size() << endl;					

	imshow("原图像", img);	
	imshow("数值调整后图像", imgResize);
	imshow("比例调整后图像", imgresize);


	
	waitKey(0);		//延迟显示
}