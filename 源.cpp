#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;
int main()
{
	VideoCapture cap(0);
	Mat frame;
	while (true) {
		cap >> frame;
		imshow("frame", frame);
		waitKey(10);
	}
	return 0;
}
