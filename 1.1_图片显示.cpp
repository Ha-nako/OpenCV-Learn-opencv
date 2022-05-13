#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void main() {

	string path = "D:/我的文件/图片/所用素材/广角镜头拍摄.jpg";	
	Mat img = imread(path);	
	namedWindow("图片", WINDOW_FREERATIO);
	imshow("图片", img);	


	waitKey(0);		
}