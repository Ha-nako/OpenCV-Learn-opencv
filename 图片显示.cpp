#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv; 

int main(int argc, char** argv)
{
	string Lu;
	Lu = "D:/�ҵ��ļ�/ͼƬ/δ�Ż���/001.jpg";
	Mat src = imread(Lu);

	if (src.empty())
	{
		printf("could not load image...\n");
		return -1;
	}

	namedWindow("Hello", WINDOW_FREERATIO);
	imshow("Hello", src);
	
	//imwrite("D:/����/11.png", src);
	//D p;
	//p.mm(src);
	waitKey(0);
	destroyWindow;
	return 0;
}