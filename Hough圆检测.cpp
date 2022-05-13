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

    img = cv::imread("D:/我的文件/图片/所用素材/调色素材/红绿蓝.jpg", 1);

    img_source = img.clone();
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY); //灰度化
    cv::GaussianBlur(img_gray, img_gray, cv::Size(3, 3), 0); //高斯滤波
    //cv::threshold(img_gray, img_gray, 70, 255, cv::THRESH_BINARY_INV);

    cv::namedWindow("example", 1);

    std::vector<cv::Vec3f> circles; //用于存储检测结果

    cv::Mat img_b;
    while (true)
    {

        //cv::dilate(img_b, img_b, cv::Mat());
        //cv::erode(img_b, img_b, cv::Mat());
        //cv::Canny(img_gray, img_b, 170, 200, 3, false); //边缘检测
        //
        //圆检测，模式1，dp=1， 最小距离20， Canny大阈值200， 圆的严格程度80， 最小半径100， 最大半径600
        cv::HoughCircles(img_gray, circles, cv::HOUGH_GRADIENT, 1, 20, 200, 80, 100, 600);

        //绘制检测到的圆
        for (size_t i = 0; i < circles.size(); i++)
        {
            //圆心
            cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);    //半径
            // 画圆心
            cv::circle(img, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // 画圆
            cv::circle(img, center, radius, cv::Scalar(0, 0, 255), 2, 8, 0);
        }

        Mat imgresize, imgResize;
        //图像不为空则显示图像
        if (img.empty() == false)
        {
            resize(img, imgresize, Size(), 0.5, 0.5);
            //resize(img_source, imgResize, Size(), 0.5, 0.5);
            cv::imshow("example", imgresize);
            //cv::imshow("binary", imgResize);
        }

        int  key = cv::waitKey(10); //等待30ms
        if (key == int('q')) //按下q退出
        {
            break;
        }


    }
    cv::destroyAllWindows(); //关闭所有窗口

    return 0;

}

