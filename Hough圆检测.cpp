#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
#include<sstream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{

    cv::Mat img, img_gray, img_source;

    img = cv::imread("D:/�ҵ��ļ�/ͼƬ/�����ز�/��ɫ�ز�/������.jpg", 1);

    img_source = img.clone();
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY); //�ҶȻ�
    cv::GaussianBlur(img_gray, img_gray, cv::Size(3, 3), 0); //��˹�˲�
    //cv::threshold(img_gray, img_gray, 70, 255, cv::THRESH_BINARY_INV);

    cv::namedWindow("example", 1);

    std::vector<cv::Vec3f> circles; //���ڴ洢�����

    cv::Mat img_b;
    while (true)
    {

        //cv::dilate(img_b, img_b, cv::Mat());
        //cv::erode(img_b, img_b, cv::Mat());
        //cv::Canny(img_gray, img_b, 170, 200, 3, false); //��Ե���
        //
        //Բ��⣬ģʽ1��dp=1�� ��С����20�� Canny����ֵ200�� Բ���ϸ�̶�80�� ��С�뾶100�� ���뾶600
        cv::HoughCircles(img_gray, circles, cv::HOUGH_GRADIENT, 1, 20, 200, 80, 100, 600);

        //���Ƽ�⵽��Բ
        for (size_t i = 0; i < circles.size(); i++)
        {
            //Բ��
            cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);    //�뾶
            // ��Բ��
            cv::circle(img, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // ��Բ
            cv::circle(img, center, radius, cv::Scalar(0, 0, 255), 2, 8, 0);
        }

        Mat imgresize, imgResize;
        //ͼ��Ϊ������ʾͼ��
        if (img.empty() == false)
        {
            resize(img, imgresize, Size(), 0.5, 0.5);
            //resize(img_source, imgResize, Size(), 0.5, 0.5);
            cv::imshow("example", imgresize);
            //cv::imshow("binary", imgResize);
        }

        int  key = cv::waitKey(10); //�ȴ�30ms
        if (key == int('q')) //����q�˳�
        {
            break;
        }


    }
    cv::destroyAllWindows(); //�ر����д���

    return 0;

}

