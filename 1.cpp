//�������
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include <iostream>
#include <string>
using namespace cv;
using namespace cv::face;
using namespace std;

void FindFaces(cv::Mat &dst)
{
Mat src = imread("src.jpg");
Mat frame = src.clone();
Mat facesRIO;
//ͼ�����ţ�����˫���Բ�ֵ��
//cv::resize(src,src,Size(128,128),0,0,cv::INTER_LINEAR);
//ͼ��ҶȻ���
cv::cvtColor(src,src,COLOR_BGR2GRAY);
//ֱ��ͼ���⻯��ͼ����ǿ�����ı��������ı䰵��
cv::equalizeHist(src,src);
//
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade,eyes_cascade;
if(!face_cascade.load(face_cascade_name))
{
    //��������������ʧ�ܣ�
    return;
}
if(!eyes_cascade.load(eyes_cascade_name))
{
    //�����۾�������ʧ�ܣ�
    return;
}
//�洢�ҵ����������Ρ�
std::vector<Rect> faces;
face_cascade.detectMultiScale(src,faces,1.1,2,0|CASCADE_SCALE_IMAGE,Size(30,30));
for(size_t i=0;i<faces.size();++i)
{
    //���ƾ��� BGR��
    rectangle(frame,faces[i],Scalar(0,0,255),1);
    //��ȡ�������ĵ㡣
    //Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    //����Բ�Ρ�
    //ellipse(frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    //��ȡ������������
    //Mat faceROI = src(faces[i]);
    //�洢�ҵ����۾����Ρ�
    //std::vector<Rect> eyes;
    //eyes_cascade.detectMultiScale(faceROI,eyes,1.1,2,0 |CASCADE_SCALE_IMAGE,Size(30,30));
    //for(size_t j=0;j<eyes.size();++j)
    //{
        //Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2,faces[i].y + eyes[j].y + eyes[j].height/2);
        //int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
        //circle(frame,eye_center,radius,Scalar( 255, 0, 0 ),4,8,0);
    //}
    //��ȡ������
    //facesROI = frame(faces[i]);
    //ͼ�����š�
    //cv::resize(facesROI,facesROI,Size(128,128),0,0,cv::INTER_LINEAR);
    //����ͼ��
    //dst = facesROI;
    //cv::imwrite("dst.jpg",facesROI);
}
return;
}