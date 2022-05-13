#include <iostream>
#include <opencv2/highgui.hpp> // ˵��˵gui ����ʲôgui �����
#include <opencv2/imgcodecs.hpp> // ͼ��ͷ�ļ�
#include <opencv2/imgproc.hpp> // ͼ����ͷ�ļ�

using namespace std;
using namespace cv;

/*Ҫ����ͼ����ò���֮ǰ
 *����Ҫ��ֵ�����ٽ����˲������ٽ���Canny��Ե���
 *�����ܼ���ͼ������
 */
Mat imgGray, imgBlur, imgCanny, imgDil;
void getContours(Mat imgDil, Mat& img);

int main()
{


    string path = "D:/�ҵ��ļ�/ͼƬ/�����ز�/��״3.jpg"; // ����ͼ�ε�ʱ����Ҫ���ұߵ����ʾ�����ļ�������
    Mat img = imread(path); // ��opencv �����е�ͼ����Ϣ��ʹ��Mat 


    // pre-processing image  ͼ��Ԥ����
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0); // ��˹�˲� 
    Canny(imgBlur, imgCanny, 25, 75);// Canny ��Ե���
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // ���� Size �� ��Ե���ķŴ����й�ϵ
    dilate(imgCanny, imgDil, kernel);

    getContours(imgDil, img); // ��һ������ ��Ѱ�������Ĳ����� �ڶ�����������ʾͼ���Ĳ���

    imshow("Image", img);
    //imshow("Image Gray", imgGray);
    //imshow("Image Blur", imgBlur);
    //imshow("Image Canny", imgCanny);
    //imshow("Image Dilate", imgDil); // ͼ��Ŵ�֮��ı�Ե���Ч��Ҫ���Ժ��� Canny ��Ե��⣬��Ҳ��Ϊʲô����������dilation��ԭ��
    waitKey(0); // ��ʱ��0���൱�������
}
// ��Ϊһ��ʼ������ͬ�����Ե���ֱ�ӽ�����Ϊ���غ���
void getContours(Mat imgDil, Mat& img)
{
    /* contour is a vector inside that vector there is more vector
     * {{Point(20,30),Point(50,60)},{},{}} each vector like a contour and each contour have some points
     *
     **/
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy; // Vec4i ���������������4�� int ����typedef    Vec<int, 4>   Vec4i;   ���ĸ�����ÿһ�㼶����һ������
    findContours(imgDil, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // CV_CHAIN_APPROX_SIMPLE - �򵥵���ʽ�ӽ���
    //drawContours(img, contours, -1, Scalar(255,0,255),2); // contouridx = -1 ������Ҫ������������������

    vector<vector<Point>> conpoly(contours.size());// conpoly(paprameter1) ,paprameter1�����vector��������������������е�vector ��ʹ����point�㼯����ֻ����ͼ�εĹսǵ㼯
    vector<Rect> boundRect(contours.size());// ��¼��ͼ�ε���Ͼ���
    string objType; // ��¼������״
    // Ϊ���˳�΢С��������˼���area �����
    for (int i = 0; i < contours.size(); i++) // ����contours.size()Ϊʲô�Ƿ��ض�ά������У���Ϊ vector::size()����ֻ����vector ����ĵ��ö�contours��������(������)��Ϊ�����
    {
        int area = contourArea(contours[i]);

        if (area > 1000)
        {
            float peri = arcLength(contours[i], true);// �ú������������ĳ��ȣ������boolֵ�������������Ƿ�պ���Ϊtrue ���������߱պ�
            //Ѱ�ҽǵ�
            // conpoly ͬ��Ϊ�����㼯�����ڶ���������ֻ��1-9������Ϊ���������������Ĺսǵ�
            approxPolyDP(contours[i], conpoly[i], 0.02 * peri, true); //  conpoly[i]�����array   0.02*peri ���������ⲻ�˾Ͳ�Ҫ��⣡���� ���һ��������Ȼ��ѯ���Ƿ�պ�
            //drawContours(img, contours , i, Scalar(255, 0, 255), 2);
             // ͨ��conpoly �����Ƶ�������ֻ���ڳ�����ΪӦ�ô��ڵĵ�
            cout << conpoly[i].size() << endl; // ���ͼ�������еĹսǵ�
            boundRect[i] = boundingRect(conpoly[i]); // ���conpoly[i] ����boundingRect �Ա�������о���
            //rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5); // ʹ��

            int objCor = (int)conpoly[i].size(); // ��������߽���
            if (3 == objCor) objType = "Triangle";
            else
                if (4 == objCor)
                {   // ����float����һ��Ҫ�ǵ�ʹ�� float ǿת����
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