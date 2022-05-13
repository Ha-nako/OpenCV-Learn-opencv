#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/////////////// Project 3 ¨C License Plate Detector //////////////////////

void main() {

	Mat img;
	VideoCapture cap(0);

	CascadeClassifier plateCascade;
	plateCascade.load("D:/RJDZ/opencv_cpp_course_resources/resources/haarcascade_russian_plate_number.xml");

	if (plateCascade.empty()) { cout << "XML file not loaded" << endl; }

	vector<Rect> plates;

	while (true) {

		cap.read(img);
		plateCascade.detectMultiScale(img, plates, 1.1, 10);//¼ì²â

		for (int i = 0; i < plates.size(); i++)
		{
			Mat imgCrop = img(plates[i]);//²ÃÇÐ
			//imshow(to_string(i), imgCrop);
			imwrite("D:/×ÀÃæ/ " + to_string(i) + ".png", imgCrop);
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);//¾ØÐÎ
		}

		imshow("Image", img);
		waitKey(1);
	}
}