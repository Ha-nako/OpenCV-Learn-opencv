#include<opencv2/opencv.hpp>
using namespace cv;

int main()
{
    Mat src, dst;
    src = imread("D:/�ҵ��ļ�/ͼƬ/�����ز�/��״2.jpg");
    if (src.empty())
    {
        printf("can not load image \n");
        return -1;
    }
    namedWindow("input", WINDOW_FREERATIO);
    imshow("input", src);
    dst = Mat::zeros(src.size(), CV_8UC3);

    blur(src, src, Size(3, 3));
    cvtColor(src, src, COLOR_BGR2GRAY);
    Canny(src, src, 20, 80, 3, false);
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    RNG rng(0);
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point(0, 0));
    }
    namedWindow("output", CV_WINDOW_AUTOSIZE);
    imshow("output", dst);
    waitKey();
    return 0;
}

/////////////////////////////////��������ͷ�ָ�////////////////////////////////////////

//#include<opencv2/imgcodecs.hpp>	//��������ͷ�ָ�
//#include<opencv2/highgui.hpp>
//#include<opencv2/imgproc.hpp>
//#include<iostream>
//
//using namespace cv;
//using namespace std;
//
//void main() {
//
//    VideoCapture cap(0);
//    //string path = "D:/�ҵ��ļ�/ͼƬ/����/QQphoto (3).jpeg";
//
//    Mat img, imgs;//= imread(path);
//    Mat imgCrop, imgmm;
//
//    Rect roi(0, 0, 640, 240);
//    //     Rect -> ���������������  ����(����x����,����y����,���ο����θ�)
//
//    Rect mm(0, 240, 640, 240);
//
//    while (1)
//    {
//        cap >> img;
//        cap >> imgs;
//
//        imgCrop = img(roi);
//        //		    �������η���ͼ����
//        imgmm = imgs(mm);
//
//        cout << "ԭͼ���С:  " << img.size() << endl;
//        cout << "�ü���ͼ���С1:" << imgCrop.size() << endl;
//        cout << "�ü���ͼ���С2:" << imgmm.size() << endl << endl;
//
//
//        imshow("ԭͼ��", img);
//        imshow("�ϰ�", imgCrop);
//        imshow("�°�", imgmm);
//
//
//        waitKey(1);
//    }
//
//
//
//
//
//}