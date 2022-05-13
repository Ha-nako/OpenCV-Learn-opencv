#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>


using namespace cv;
using namespace std;

int alpha_slider1, alpha_slider2;
void on_trackbar(int, void*) {}

int main()
{
	VideoCapture capture(0);	// 从摄像头读入视频
	namedWindow("[总]");
	createTrackbar("Hmin", "[总]", &alpha_slider1, 180, on_trackbar);
	createTrackbar("Hmax", "[总]", &alpha_slider2, 180, on_trackbar);
	Mat frame; // 定义一个Mat变量，用于存储每一帧的图像
	Mat HSV;
	Mat mask; // 保存inRange后的值
	while (1)
	{
		capture >> frame;  // 读取当前帧    
		cvtColor(frame, HSV, COLOR_BGR2HSV);	// 转化为HSV					
		inRange(HSV, Scalar(alpha_slider1, 43, 46), Scalar(alpha_slider2, 255, 255), mask);
		imshow("保存", mask);
		imshow("[总]", frame);
		waitKey(30); // 延时30ms
	}
	return 0;
}