#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  
#include<math.h>
#include<opencv2/opencv.hpp>


using namespace cv;
using namespace std;
using namespace ml;


ostringstream oss;

int num = -1;
Mat dealimage;
Mat src;
int k = 0;
Mat yangben_gray;
Mat yangben_thresh;


int main() {

	//ͼƬ��ȡ·��
	string path = "D://����//YY_2.jpg";
	Mat img = imread(path);
	//imshow("ԭͼ", img);

	if (img.empty())
	{
		cerr << "δ�ҵ��ļ�������" << endl;
		return -1;
	}

	//============================== = ��ȡѵ������============================== =

	const int classsum = 3;//ͼƬ����6��
	const int imagesSum = 82;//ÿ����82��ͼƬ			   
	const int imageRows = 44;//ͼƬ�ߴ�
	const int imageCols = 44;
	//ѵ�����ݣ�ÿһ��һ��ѵ��ͼƬ
	Mat trainingData;
	//ѵ��������ǩ
	Mat labels;
	//���յ�ѵ��������ǩ
	Mat clas;
	//���յ�ѵ������
	Mat traindata;

	//��ָ���ļ�������ȡͼƬ//
	for (int p = 0; p < classsum; p++)
	{
		//ѵ������ȡ·��
		oss << "D://����//ѵ��//";
		num += 1;//num��0��2
		int label = num;
		oss << num << "//*.png";	//ͼƬ���ֺ�׺��oss���Խ���������ַ���
		string pattern = oss.str();	//oss.str()���oss�ַ��������Ҹ���pattern
		oss.str("");				//ÿ��ѭ�����oss�ַ������
		vector<Mat> input_images;
		vector<String> input_images_name;
		glob(pattern, input_images_name, false);
		//Ϊfalseʱ����������ָ���ļ����ڷ���ģʽ���ļ�����Ϊtrueʱ����ͬʱ����ָ���ļ��е����ļ���
		//��ʱinput_images_name��ŷ���������ͼƬ��ַ

		int all_num = input_images_name.size();//�ļ����ܹ��м���ͼƬ
		cout << num << ":�ܹ���" << all_num << "��ͼƬ������" << endl;
		cout << "-------------------------------" << endl;

		for (int i = 0; i < imagesSum; i++)
		{
			cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);
			threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);
			input_images.push_back(yangben_thresh);
			//ѭ����ȡÿ��ͼƬ�������η���vector<Mat> input_images��
			dealimage = input_images[i];

			//����reshape()�������������ȡ,
			//reshape(1, 1)�Ľ������ԭͼ���Ӧ�ľ��󽫱������һ��һ�е���������Ϊ����������
			dealimage = dealimage.reshape(1, 1);//ͼƬ���л�
			trainingData.push_back(dealimage);//���л����ͼƬ���δ���
			labels.push_back(label);//��ÿ��ͼƬ��Ӧ�ı�ǩ���δ���
		}
	}

	//ͼƬ���ݺͱ�ǩת����
	Mat(trainingData).copyTo(traindata);//����
	traindata.convertTo(traindata, CV_32FC1);//����ͼƬ���ݵ����ͣ���Ҫ����Ȼ�����
	Mat(labels).copyTo(clas);//����


	//============================== = ����KNNģ��============================== =

	Ptr<KNearest>knn = KNearest::create();
	knn->setDefaultK(10);//k�������
	knn->setIsClassifier(true);//trueΪ���࣬falseΪ�ع�
	//ѵ�����ݺͱ�ǩ�Ľ��
	Ptr<TrainData>trainData = TrainData::create(traindata, ROW_SAMPLE, clas);
	//ѵ��
	knn->train(trainData);

	//model->save("D://ѵ��ģ��//KNearestModel.xml"); 


//������������������������������������������������������������������������������


	Mat imggray, imgblur, imgCanny, imgDil;	//����ͼ�����

	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//��ת���Ҷ�ͼ��:1ת��Ϊ2,ģʽΪ3���Ҷȣ�
	//imshow("�Ҷ�", imggray);

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//����˹ģ��:size(�����ں˴�С7*7),�����Ļ�ϵ�λ��ƫ��->5,0(x,y)

	Canny(imgblur, imgCanny, 30, 100);
	//��������ؼ��	  ���������ֵ(�ɸ���,ֵԽС��ԵԽ��)

	//imshow("ԭͼ", img);
	//imshow("������ؼ��", imgCanny);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��)
	dilate(imgCanny, imgDil, kernel);
	//��	��Ե����
	namedWindow("��Ե����", WINDOW_FREERATIO);
	imshow("��Ե����", imgDil);

	std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ

		//����Բ
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 60, 80, 120);
	//	
	Mat imgCrop, hist;


	//+++++++++++++++++++++++++ð������1+++++++++++++++++++++++++++++++
	//int t, tt;
	//for (int i = 0; i < 96; i++)
	//{
	//	for (int j = 0; j < 96 - i - 1; j++)
	//	{
	//		if (circles[j][0] > circles[j + 1][0])
	//		{
	//			t = circles[j][0];
	//			tt = circles[j][1];

	//			circles[j][0] = circles[j + 1][0];
	//			circles[j][1] = circles[j + 1][1];

	//			circles[j + 1][0] = t;
	//			circles[j + 1][1] = tt;

	//		}
	//	}
	//}

	////-------------------
	//int k = 0, s = 0;

	//for (k = 0; k < 12; k++)
	//{
	//	for (int i = 0; i < 8; i++)
	//	{
	//		for (int j = 1; j < 8 - i; j++)
	//		{

	//			//if (circles[j ][1] > circles[j + 1 ][1])
	//			if (circles[k * 8 + i][1] > circles[k * 8 + i + j][1])
	//			{
	//				t = circles[k * 8 + i][1];
	//				tt = circles[k * 8 + i][0];

	//				circles[k * 8 + i][0] = circles[k * 8 + i + j][0];
	//				circles[k * 8 + i][1] = circles[k * 8 + i + j][1];

	//				circles[k * 8 + i + j][1] = t;
	//				circles[k * 8 + i + j][0] = tt;
	//			}
	//			//cout << endl << "j " << j  << endl;
	//		}

	//	}
	//}
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	//��������Բ��
	for (size_t i = 0; i < circles.size(); i++)
	{
		cout << endl << "�� " << i + 1 << " ��Բ��" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//����Բ����  
		//circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);

		std::cout << "Բ�İ뾶��" << radius << std::endl;
		std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;
		cout << "----------------------------------------" << endl;

		Rect roi(circles[i][0] - 22, circles[i][1] - 22, 44, 44);
		//     Rect -> ���������������  ����(����x����,����y����,���ο����θ�)
		Rect roi_2(circles[i][0] - 2.5, circles[i][1] - 2.5, 5, 5);

		//rectangle(img, Point(circles[i][0] - 5, circles[i][1] - 5), Point(circles[i][0] +5, circles[i][1] + 5), Scalar(255, 255, 255), FILLED);
		//				���������Ͻ�	�����½�


		imgCrop = imggray(roi);
		//		    �������η���ͼ����
		Mat imgCrop_2 = imggray(roi_2);
		//imshow("2", imgCrop_2);
		Mat imgCrop_3 = img(roi);
		//imshow("imgCrop_3", imgCrop_3);

	//==============================Ԥ�����============================== =

		cvtColor(imgCrop_3, imgCrop_3, COLOR_BGR2GRAY);
		threshold(imgCrop_3, imgCrop_3, 0, 255, CV_THRESH_OTSU);
		//imshow("ԭͼ��", img);
		Mat input;
		imgCrop_3 = imgCrop_3.reshape(1, 1);//����ͼƬ���л�
		input.push_back(imgCrop_3);
		input.convertTo(input, CV_32FC1);//����ͼƬ���ݵ����ͣ���Ҫ����Ȼ�����

		float r = knn->predict(input);   //�������н���Ԥ��
		//cout << r << endl;

		int rr = (int)r;

		//����  ����������ͼƬ
		switch (rr)
		{
		case 0:
			putText(img, "0", Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 1.2);
			break;
		case 1:
			putText(img, "2", Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 0, 0), 1.2);
			break;
		case 2:
			putText(img, "4", Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 1.2);
			break;

		default:
			break;
		}

		//&&&&&&&&&&&&&&&&&&-----�ɼ�����-----&&&&&&&&&&&&&&&&&		
		//for (int s = 0; s < 6; s++)
		//{
		//	
		//}
		//imwrite("D://����//�½��ļ���//" + to_string(i-16) + ".png", imgCrop_3);
		//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		//----------------------------------------------------------------------------------------------

	}
	imshow("��Ч��ͼ��", img);
	waitKey(0);

	return 0;
}

