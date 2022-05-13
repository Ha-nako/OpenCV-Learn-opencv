#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  
#include<fstream>		//�ļ�����ͷ�ļ�
#include<math.h>
#include<opencv2/opencv.hpp>


//ofstream			д�ļ�
//ifstream			���ļ�
//fstream			��д�ļ�

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





int getHistograph(const Mat grayImage);
void GetGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev);
int Lei = 0;
int sum_Dian = 0;
int sum_allTD = 0;

int main() {

	string path = "D://pp//yy.bmp";
	Mat img = imread(path);	//���ļ�
	//imshow("ԭͼ", img);


	//cout << "д�ļ���" << endl;
	ofstream ofs;		//����������(д�ļ�)

	ofs.open("D://����//01.txt", ios::trunc);

	double a, b;

	if (img.empty())
	{
		cerr << "δ�ҵ��ļ�������" << endl;
		return -1;
	}


	//============================== = ��ȡѵ������============================== =
	//const int classsum = 3;//ͼƬ����6��
	//const int imagesSum = 64;//ÿ����64��ͼƬ			   
	//const int imageRows = 44;//ͼƬ�ߴ�
	//const int imageCols = 44;
	////ѵ�����ݣ�ÿһ��һ��ѵ��ͼƬ
	//Mat trainingData;
	////ѵ��������ǩ
	//Mat labels;
	////���յ�ѵ��������ǩ
	//Mat clas;
	////���յ�ѵ������
	//Mat traindata;
	////��ָ���ļ�������ȡͼƬ//
	////cout << 12313213213213 << endl;
	//for (int p = 0; p < classsum; p++)
	//{
	//	oss << "D://����//ѵ��//";
	//	num += 1;//num��0��5
	//	int label = num;
	//	oss << num << "//*.png";//ͼƬ���ֺ�׺��oss���Խ���������ַ���
	//	string pattern = oss.str();//oss.str()���oss�ַ��������Ҹ���pattern
	//	oss.str("");//ÿ��ѭ�����oss�ַ������
	//	vector<Mat> input_images;
	//	vector<String> input_images_name;
	//	glob(pattern, input_images_name, false);
	//	//Ϊfalseʱ����������ָ���ļ����ڷ���ģʽ���ļ�����Ϊtrueʱ����ͬʱ����ָ���ļ��е����ļ���
	//	//��ʱinput_images_name��ŷ���������ͼƬ��ַ
	//	int all_num = input_images_name.size();//�ļ����ܹ��м���ͼƬ
	//	cout << num << ":�ܹ���" << all_num << "��ͼƬ������" << endl;

	//	for (int i = 0; i < imagesSum; i++)
	//	{
	//		cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);
	//		threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);
	//		input_images.push_back(yangben_thresh);
	//		//ѭ����ȡÿ��ͼƬ�������η���vector<Mat> input_images��
	//		dealimage = input_images[i];


	//		//ע�⣺���Ǽ򵥴ֱ�������ͼ������������Ϊ����������Ϊ���ǹ�ע�������������ѵ������
	//		//������ѡ������򵥵ķ�ʽ���������ȡ�������������⣬
	//		//������ȡ�ķ�ʽ�кܶ࣬����LBP��HOG�ȵ�
	//		//��������reshape()�������������ȡ,
	//		//reshape(1, 1)�Ľ������ԭͼ���Ӧ�ľ��󽫱������һ��һ�е���������Ϊ����������
	//		dealimage = dealimage.reshape(1, 1);//ͼƬ���л�
	//		trainingData.push_back(dealimage);//���л����ͼƬ���δ���
	//		labels.push_back(label);//��ÿ��ͼƬ��Ӧ�ı�ǩ���δ���
	//	}
	//}

	////ͼƬ���ݺͱ�ǩת����
	//Mat(trainingData).copyTo(traindata);//����
	//traindata.convertTo(traindata, CV_32FC1);//����ͼƬ���ݵ����ͣ���Ҫ����Ȼ�����
	//Mat(labels).copyTo(clas);//����


	////============================== = ����KNNģ��============================== =
	//Ptr<KNearest>knn = KNearest::create();
	//knn->setDefaultK(10);//k�������
	//knn->setIsClassifier(true);//trueΪ���࣬falseΪ�ع�
	////ѵ�����ݺͱ�ǩ�Ľ��
	//Ptr<TrainData>trainData = TrainData::create(traindata, ROW_SAMPLE, clas);
	////ѵ��
	//knn->train(trainData);

	////model->save("D://ѵ��ģ��//KNearestModel.xml"); 


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

	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	//�������ʹ������(����/��ʴ)���ں�  (��ԽС,����Խ��)
	dilate(imgCanny, imgDil, kernel);
	//��	��Ե����
	//imshow("��Ե����", imgDil);

	std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ

		//����Բ
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//	
	Mat imgCrop, hist;


	//+++++++++++++++++++++++++ð������1+++++++++++++++++++++++++++++++
	int t, tt;
	for (int i = 0; i < 96; i++)
	{
		for (int j = 0; j < 96 - i - 1; j++)
		{
			if (circles[j][0] > circles[j + 1][0])
			{
				t = circles[j][0];
				tt = circles[j][1];

				circles[j][0] = circles[j + 1][0];
				circles[j][1] = circles[j + 1][1];

				circles[j + 1][0] = t;
				circles[j + 1][1] = tt;

			}
		}
	}



	//-------------------
	int k = 0, s = 0;

	for (k = 0; k < 12; k++)
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = 1; j < 8 - i; j++)
			{

				//if (circles[j ][1] > circles[j + 1 ][1])
				if (circles[k * 8 + i][1] > circles[k * 8 + i + j][1])
				{
					t = circles[k * 8 + i][1];
					tt = circles[k * 8 + i][0];

					circles[k * 8 + i][0] = circles[k * 8 + i + j][0];
					circles[k * 8 + i][1] = circles[k * 8 + i + j][1];

					circles[k * 8 + i + j][1] = t;
					circles[k * 8 + i + j][0] = tt;
				}
				//cout << endl << "j " << j  << endl;
			}

		}
	}
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	for (size_t i = 1; i < 2; i++)
	{
		cout << endl << "�� " << i + 1 << " ��Բ��" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//����Բ����  
		//circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);

		//int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
		//int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
		//int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
		//std::cout << "Բ�İ뾶��" << radius << std::endl;
		//std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std::endl;


		//Mat image = imread(IMG_PATH);
		//Mat dst1 = Mat::zeros(img.size(), img.type());
		//Mat mask = Mat::zeros(img.size(), CV_8U);



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



		//###################################
		//Ԥ�����
		////Mat src = imgCrop_3;
		//cvtColor(imgCrop_3, imgCrop_3, COLOR_BGR2GRAY);

		//threshold(imgCrop_3, imgCrop_3, 0, 255, CV_THRESH_OTSU);
		////imshow("ԭͼ��", img);
		//Mat input;
		//imgCrop_3 = imgCrop_3.reshape(1, 1);//����ͼƬ���л�
		//input.push_back(imgCrop_3);
		//input.convertTo(input, CV_32FC1);//����ͼƬ���ݵ����ͣ���Ҫ����Ȼ�����

		//float r = knn->predict(input);   //�������н���Ԥ��
		//cout << r << endl;

		//int rr = (int)r;

		//if (rr == 2)
		//{
		//	putText(img, "4", Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 255), 1);

		//}
		//if (rr == 0)
		//{
		//	putText(img, "0", Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 255), 1);

		//}
		//if (rr == 1)
		//{
		//	putText(img, "2", Point(circles[i][0] - 10, circles[i][1] + 10), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 255), 1);

		//}


		//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&����

		for (int s = 0; s < 18; s++)
		{
			imwrite("D://����//�½��ļ���//" + to_string(s + 64) + ".png", imgCrop_3);
		}
		
		//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		Mat dst = Mat(imgCrop.size(), CV_8UC1);
		Mat dst1 = Mat(imgCrop.size(), CV_8UC3);



		//*****************************************************************
		//*****************************************************************

		//int Zhi = getHistograph(imgCrop,imgCrop_2);
		int Zhi;
		int sum0 = 0;
		int sum1 = 0;
		int sum2 = 0;
		int sum3 = 0;
		sum_allTD = 0;

		//��������++++++++++++++++++++++++++++++++++++

		//����ͼ�����ط���:ͨ��at<type>(i,j)����ָ��
		int num_s = 0;
		for (int i = 0; i < imgCrop.rows; i++)
		{
			for (int j = 0; j < imgCrop.cols; j++)
			{
				if (imgCrop.channels() == 1)
				{
					dst.at<uchar>(i, j) = imgCrop.at<uchar>(i, j) + 1;

					if (sqrt(i * i + j * j) < radius)
					{
						num_s = num_s + 1;
					}
				}
				//else if (imgCrop.channels() == 3)
				//{
				//	dst1.at<Vec3b>(i, j)[0] = imgCrop.at<Vec3b>(i, j)[0] + 100;
				//	dst1.at<Vec3b>(i, j)[1] = imgCrop.at<Vec3b>(i, j)[1] + 100;
				//	dst1.at<Vec3b>(i, j)[2] = imgCrop.at<Vec3b>(i, j)[2] + 100;
				//	
				//}
			}
		}
		cout << "-----------" << endl << "num_s===" << num_s << endl;



		double a = 0, b = 0;

		//�ҶȾ�ֵ������---------------------------
		GetGrayAvgStdDev(imgCrop_2, a, b);

		int aa = (int)a;
		int sum_DanTongD_zhi = aa - num_s;

		if (sum_DanTongD_zhi < 0)
		{
			sum_DanTongD_zhi = sum_DanTongD_zhi * (-1);
		}
		Zhi = sum_DanTongD_zhi;



		//������ֱ��ͼ��ͨ����Ŀ����0��ʼ����
		int channels[] = { 0 };
		//����ֱ��ͼ����ÿһά�ϵĴ�С������Ҷ�ͼֱ��ͼ�ĺ�������ͼ��ĻҶ�ֵ����һά��bin�ĸ���
		//���ֱ��ͼͼ�������bin����Ϊx��������bin����Ϊy����channels[]={1,2}��ֱ��ͼӦ��Ϊ��ά�ģ�Z����ÿ��bin��ͳ�Ƶ���Ŀ
		const int histSize[] = { 256 };
		//ÿһάbin�ı仯��Χ
		float range[] = { 0,256 };

		//����bin�ı仯��Χ��������channelsӦ�ø�channelsһ��
		const float* ranges[] = { range };

		//����ֱ��ͼ�����������ֱ��ͼ����
		Mat hist;
		//opencv�м���ֱ��ͼ�ĺ�����hist��СΪ256*1��ÿ�д洢��ͳ�Ƶĸ��ж�Ӧ�ĻҶ�ֵ�ĸ���
		calcHist(&imgCrop, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);//cv����cvCalcHist

		//�ҳ�ֱ��ͼͳ�Ƶĸ��������ֵ��������Ϊֱ��ͼ������ĸ�
		double maxValue = 0;
		//�Ҿ����������Сֵ����Ӧ�����ĺ���
		minMaxLoc(hist, 0, &maxValue, 0, 0);
		//���ֵȡ��
		int rows = cvRound(maxValue);
		//����ֱ��ͼͼ��ֱ��ͼ������ĸ���Ϊ����������Ϊ256(�Ҷ�ֵ�ĸ���)
		//��Ϊ��ֱ��ͼ��ͼ�������Ժڰ���ɫΪ���֣���ɫΪֱ��ͼ��ͼ��
		Mat histImage = Mat::zeros(rows, 256, CV_8UC1);

		//ֱ��ͼͼ���ʾ
		for (int i = 0; i < 256; i++)
		{
			//ȡÿ��bin����Ŀ
			int temp = (int)(hist.at<float>(i, 0));
			//���bin��ĿΪ0����˵��ͼ����û�иûҶ�ֵ��������Ϊ��ɫ
			//���ͼ�����иûҶ�ֵ���򽫸��ж�Ӧ������������Ϊ��ɫ
			if (temp)
			{
				//����ͼ�������������Ͻ�Ϊԭ�㣬����Ҫ���б任��ʹֱ��ͼͼ�������½�Ϊ����ԭ��
				histImage.col(i).rowRange(Range(rows - temp, rows)) = 255;
			}




			//if (150 < i < 200)
			//{
			//	sum2 = sum2 + temp;
			//}
			if (i > 155)
			{
				sum3 = sum3 + temp;
			}
			if (i > 200)
			{
				sum1 = sum1 + temp;
			}
			//sum_Dian += sum_Dian;
			//sum_allTD += sum_DanTongD_zhi;

		}
		//�ж�ͨ�������ص����
		if (sum1 > 100)
		{
			Lei = 1;
		}
		else if (sum3 == 0)
		{
			Lei = 3;
		}
		else
		{
			Lei = 2;
		}

		//����ֱ��ͼͼ���и߿��ܸܺߣ���˽���ͼ�����Ҫ���ж�Ӧ��������ʹֱ��ͼͼ���ֱ��
		Mat resizeImage;
		resize(histImage, resizeImage, Size(256, 256));
		//return resizeImage;
		//return sum_allTD;
		//Zhi = sum_allTD;

		//************************
		//***********************
		//***************************




		//GetGrayAvgStdDev(imgCrop, a, b);
		//int bb = (int)b;
		//int aa = (int)a;
		////****����
		//aa-sum_Dian


		//��ӡֵ
		putText(img, to_string(sum_DanTongD_zhi), Point(circles[i][0] - 10, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 0, 0), 0.5);


		//cout << "----------------------" << endl << Zhi << endl;

		//----------------------------------------------------------------------------------------------

		//int mm = 1;	
		//if ((i % 8) == 0)
		//{
		//	ofs << "-----------------" << endl;
		//	ofs << "���� " << mm << " �С���" << endl;
		//	ofs << "-----------------" << endl;
		//	mm++;
		//}

		ofs << "���� " << i + 1 << " ��Բ����" << endl;

		//ofs << "Բ��X��" << circles[i][0] << ",Բ��Y��" << circles[i][1] << std::endl;
		ofs << "����ó�����ֵ��" << Zhi << endl << endl;

		//putText(img, to_string((int)circles[i][0]), Point(circles[i][0] - 5, circles[i][1] -5), FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 0, 0), 1);
		//putText(img, to_string((int)circles[i][1]), Point(circles[i][0] -5, circles[i][1]+5), FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 0, 0), 1);
		//putText(img, to_string(Zhi), Point(circles[i][0] - 18, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 0.4, Scalar(0, 0, 255), 1);


		//----------------------------------------------------------------------------------------------


		//����
		//if (Lei == 1)
		//{
		//	putText(img, "1", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	���֣�	  ������					       �����壨�����		������		      ��ȡ�
		//}
		//if (Lei == 2)
		//{
		//	putText(img, "2", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	���֣�	  ������					       �����壨�����		������		      ��ȡ�
		//}
		//if (Lei == 3)
		//{
		//	putText(img, "3", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	���֣�	  ������					       �����壨�����		������		      ��ȡ�
		//}
		//Lei = 0;
		//sum_Dian = 0;

		//imshow("hist"+i, hist);
		//imshow("����ͼ��+"+i, imgCrop);

	}
	imshow("��Ч��ͼ��", img);
	waitKey(0);		//�ӳ���ʾ

	ofs.close();
	return 0;
}

//int getHistograph( Mat grayImage,Mat imgppp)
//{
//	
//}

//����ҶȾ�ֵ������
void GetGrayAvgStdDev(cv::Mat& src, double& avg, double& stddev)
{
	cv::Mat img;
	if (src.channels() == 3)
		cv::cvtColor(src, img, CV_BGR2GRAY);
	else
		img = src;
	cv::mean(src);
	cv::Mat mean;
	cv::Mat stdDev;
	cv::meanStdDev(img, mean, stdDev);

	avg = mean.ptr<double>(0)[0];
	stddev = stdDev.ptr<double>(0)[0];
}



