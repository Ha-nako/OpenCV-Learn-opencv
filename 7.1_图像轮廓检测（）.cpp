#include <iostream>
#include <opencv2/highgui.hpp> // 说是说gui 具体什么gui 不清楚
#include <opencv2/imgcodecs.hpp> // 图像头文件
#include <opencv2/imgproc.hpp> // 图像处理头文件

using namespace std;
using namespace cv;

/*要进行图像形貌检测之前
 *首先要二值化，再进行滤波处理，再进行Canny边缘检测
 *最后才能检测出图形轮廓
 */
Mat imgGray, imgBlur, imgCanny, imgDil;
void getContours(Mat imgDil, Mat& img);

int main()
{


    string path = "D:/我的文件/图片/所用素材/形状3.jpg"; // 导入图形的时候，先要在右边点击显示所有文件！！！
    Mat img = imread(path); // 在opencv 中所有的图像信息都使用Mat 


    // pre-processing image  图像预处理
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0); // 高斯滤波 
    Canny(imgBlur, imgCanny, 25, 75);// Canny 边缘检测
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 其中 Size 和 边缘检测的放大倍数有关系
    dilate(imgCanny, imgDil, kernel);

    getContours(imgDil, img); // 第一个参数 是寻找轮廓的参数， 第二个参数是显示图案的参数

    imshow("Image", img);
    //imshow("Image Gray", imgGray);
    //imshow("Image Blur", imgBlur);
    //imshow("Image Canny", imgCanny);
    //imshow("Image Dilate", imgDil); // 图像放大之后的边缘检测效果要明显好于 Canny 边缘检测，这也是为什么大佬热衷于dilation的原因
    waitKey(0); // 延时，0即相当于无穷大
}
// 因为一开始参数不同，所以电脑直接将其视为重载函数
void getContours(Mat imgDil, Mat& img)
{
    /* contour is a vector inside that vector there is more vector
     * {{Point(20,30),Point(50,60)},{},{}} each vector like a contour and each contour have some points
     *
     **/
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy; // Vec4i 即代表该向量内有4个 int 变量typedef    Vec<int, 4>   Vec4i;   这四个向量每一层级代表一个轮廓
    findContours(imgDil, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // CV_CHAIN_APPROX_SIMPLE - 简单的链式接近法
    //drawContours(img, contours, -1, Scalar(255,0,255),2); // contouridx = -1 代表需要绘制所检测的所有轮廓

    vector<vector<Point>> conpoly(contours.size());// conpoly(paprameter1) ,paprameter1便代表vector对象的行数，而其列数中的vector 是使用了point点集但其只包含图形的拐角点集
    vector<Rect> boundRect(contours.size());// 记录各图形的拟合矩形
    string objType; // 记录物体形状
    // 为了滤除微小噪声，因此计算area 的面积
    for (int i = 0; i < contours.size(); i++) // 关于contours.size()为什么是返回二维数组的行，因为 vector::size()函数只接受vector 对象的调用而contours的所有行(不管列)均为其对象
    {
        int area = contourArea(contours[i]);

        if (area > 1000)
        {
            float peri = arcLength(contours[i], true);// 该函数计算轮廓的长度，后面的bool值表面轮廓曲线是否闭合若为true 则轮廓曲线闭合
            //寻找角点
            // conpoly 同样为轮廓点集但它第二个数组中只有1-9个参数为了描述各个轮廓的拐角点
            approxPolyDP(contours[i], conpoly[i], 0.02 * peri, true); //  conpoly[i]是输出array   0.02*peri 这个参数理解不了就不要理解！！！ 最后一个参数仍然是询问是否闭合
            //drawContours(img, contours , i, Scalar(255, 0, 255), 2);
             // 通过conpoly 而绘制的轮廓中只存在程序认为应该存在的点
            cout << conpoly[i].size() << endl; // 输出图像轮廓中的拐角点
            boundRect[i] = boundingRect(conpoly[i]); // 针对conpoly[i] 进行boundingRect 以便拟合相切矩形
            //rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5); // 使用

            int objCor = (int)conpoly[i].size(); // 计算物体边角数
            if (3 == objCor) objType = "Triangle";
            else
                if (4 == objCor)
                {   // 计算float对象，一定要记得使用 float 强转符号
                    float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                    if (aspRatio < 1.05 && aspRatio>0.95)
                        objType = "Square";
                    else objType = "Rectangle";
                }
                else if (objCor > 4) objType = "Circle";

            putText(img, objType, Point(boundRect[i].x, boundRect[i].y - 5), FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 1);
            drawContours(img, conpoly, i, Scalar(255, 0, 255), 2);
            rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
        }


    }
}