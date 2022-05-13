#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

Mat img, imgGray, imgBlur, imgCanny, imgThre, imgDil, imgErode, imgWarp, imgCrop;

void main() {

	string path = "D:/RJDZ/opencv_cpp_course_resources/resources/paper.jpg";
	Mat img = imread(path);
	resize(img, img, Size(), 0.5, 0.5);

	imshow("Image", img);


	waitKey(0);
}