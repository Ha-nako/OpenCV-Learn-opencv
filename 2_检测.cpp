//#include "mainwindow.h"
//#include <QApplication>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include<opencv2/face.hpp>
using namespace  cv;
using namespace cv::face;
using namespace std;

int main(int argc, char* argv[])
{
    VideoCapture v(0);
    Mat src;
    namedWindow("src", CV_WINDOW_AUTOSIZE);
    string filename = "testrcm.csv";
    ifstream file(filename.c_str());
    //if (file == NULL) {
    //    cout << "file NULL" << endl;
    //    return -1;
    //}
    //else {
    //    cout << "file ok" << endl;
    //}
    string line;
    string path;
    char op = ';';
    string classLabel;
    vector<Mat> images;
    vector<int> labels;
    while (getline(file, line)) {
        cout << "line:" << line << endl;
        stringstream liness(line);
        getline(liness, path, op);
        cout << "path : " << path << endl;
        getline(liness, classLabel);
        cout << "class Label : " << classLabel << endl;
        if ((path.empty() == 0) && (classLabel.empty() == 0)) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classLabel.c_str()));
        }

    }
    char c;
    cout << "image : size : " << images.size() << " labels : size : " << labels.size() << endl;
    Mat testS = images[images.size() - 1];
    int testI = labels[labels.size() - 1];
    cout << "test id : " << testI << endl;
    images.pop_back();
    labels.pop_back();
    cout << "images : size : " << images.size() << endl;
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    Mat dst;
    Mat dst1;
    vector<Rect> faces;
    int i = 0;
    Mat image;
    Mat imagel;
    int count = 1;
    CascadeClassifier c1;
    bool res = c1.load("D:/RJDZ/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");

    while (v.read(src)) {
        dst = Mat::zeros(src.size(), src.type());
        cvtColor(src, dst, CV_BGR2GRAY);
        dst1 = Mat::zeros(dst.size(), dst.type());
        equalizeHist(dst, dst1);
        c1.detectMultiScale(dst1, faces, 1.1, 3, 0, Size(50, 50));
        cout << "num :" << faces.size() << endl;
        for (i = 0; i < faces.size(); i++)
        {
            rectangle(src, faces[i], Scalar(0, 0, 255), 2, LINE_8, 0);
            image = src(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
            resize(image, imagel, Size(100, 100));
            cvtColor(imagel, imagel, CV_BGR2GRAY);
            cout << "imagel.type : " << imagel.type() << " imagel.size : " << imagel.size() << endl;
            cout << "testS.type : " << testS.type() << " testS.size : " << testS.size << endl;
            int resId = 0;
            resId = model->predict(imagel);
            cout << "testId : " << resId << endl;
            if (resId == 1)
            {
                putText(src, "Trump", Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 2, LINE_8);
            }
            if (resId == 10)
            {
                putText(src, "musk", Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 2, LINE_8);
            }
            if (resId == 100)
            {
                putText(src, "lee", Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 2, LINE_8);
            }
            count++;
        }
        imshow("src", src);
        c = waitKey(30);
        if (c == 27)
        {
            break;
        }

    }
    waitKey(0);
    return 0;
}
