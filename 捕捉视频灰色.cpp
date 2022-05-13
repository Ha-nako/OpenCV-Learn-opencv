#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>


using namespace cv;
using namespace std;

int alpha_slider1, alpha_slider2;
void on_trackbar(int, void*) {}

int main()
{
	VideoCapture capture(0);	// ������ͷ������Ƶ
	namedWindow("[��]");
	createTrackbar("Hmin", "[��]", &alpha_slider1, 180, on_trackbar);
	createTrackbar("Hmax", "[��]", &alpha_slider2, 180, on_trackbar);
	Mat frame; // ����һ��Mat���������ڴ洢ÿһ֡��ͼ��
	Mat HSV;
	Mat mask; // ����inRange���ֵ
	while (1)
	{
		capture >> frame;  // ��ȡ��ǰ֡    
		cvtColor(frame, HSV, COLOR_BGR2HSV);	// ת��ΪHSV					
		inRange(HSV, Scalar(alpha_slider1, 43, 46), Scalar(alpha_slider2, 255, 255), mask);
		imshow("����", mask);
		imshow("[��]", frame);
		waitKey(30); // ��ʱ30ms
	}
	return 0;
}