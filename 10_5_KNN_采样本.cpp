#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <thread>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  
#include<fstream>		//文件操作头文件
#include<math.h>
#include<opencv2/opencv.hpp>


//ofstream			写文件
//ifstream			读文件
//fstream			读写文件

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
	Mat img = imread(path);	//读文件
	//imshow("原图", img);


	//cout << "写文件：" << endl;
	ofstream ofs;		//创建流对象(写文件)

	ofs.open("D://桌面//01.txt", ios::trunc);

	double a, b;

	if (img.empty())
	{
		cerr << "未找到文件！！！" << endl;
		return -1;
	}


	//============================== = 读取训练数据============================== =
	//const int classsum = 3;//图片共有6类
	//const int imagesSum = 64;//每类有64张图片			   
	//const int imageRows = 44;//图片尺寸
	//const int imageCols = 44;
	////训练数据，每一行一个训练图片
	//Mat trainingData;
	////训练样本标签
	//Mat labels;
	////最终的训练样本标签
	//Mat clas;
	////最终的训练数据
	//Mat traindata;
	////从指定文件夹下提取图片//
	////cout << 12313213213213 << endl;
	//for (int p = 0; p < classsum; p++)
	//{
	//	oss << "D://桌面//训练//";
	//	num += 1;//num从0到5
	//	int label = num;
	//	oss << num << "//*.png";//图片名字后缀，oss可以结合数字与字符串
	//	string pattern = oss.str();//oss.str()输出oss字符串，并且赋给pattern
	//	oss.str("");//每次循环后把oss字符串清空
	//	vector<Mat> input_images;
	//	vector<String> input_images_name;
	//	glob(pattern, input_images_name, false);
	//	//为false时，仅仅遍历指定文件夹内符合模式的文件，当为true时，会同时遍历指定文件夹的子文件夹
	//	//此时input_images_name存放符合条件的图片地址
	//	int all_num = input_images_name.size();//文件下总共有几个图片
	//	cout << num << ":总共有" << all_num << "个图片待测试" << endl;

	//	for (int i = 0; i < imagesSum; i++)
	//	{
	//		cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);
	//		threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);
	//		input_images.push_back(yangben_thresh);
	//		//循环读取每张图片并且依次放在vector<Mat> input_images内
	//		dealimage = input_images[i];


	//		//注意：我们简单粗暴将整个图的所有像素作为了特征，因为我们关注更多的是整个的训练过程
	//		//，所以选择了最简单的方式完成特征提取工作，除此中外，
	//		//特征提取的方式有很多，比如LBP，HOG等等
	//		//我们利用reshape()函数完成特征提取,
	//		//reshape(1, 1)的结果就是原图像对应的矩阵将被拉伸成一个一行的向量，作为特征向量。
	//		dealimage = dealimage.reshape(1, 1);//图片序列化
	//		trainingData.push_back(dealimage);//序列化后的图片依次存入
	//		labels.push_back(label);//把每个图片对应的标签依次存入
	//	}
	//}

	////图片数据和标签转变下
	//Mat(trainingData).copyTo(traindata);//复制
	//traindata.convertTo(traindata, CV_32FC1);//更改图片数据的类型，必要，不然会出错
	//Mat(labels).copyTo(clas);//复制


	////============================== = 创建KNN模型============================== =
	//Ptr<KNearest>knn = KNearest::create();
	//knn->setDefaultK(10);//k个最近领
	//knn->setIsClassifier(true);//true为分类，false为回归
	////训练数据和标签的结合
	//Ptr<TrainData>trainData = TrainData::create(traindata, ROW_SAMPLE, clas);
	////训练
	//knn->train(trainData);

	////model->save("D://训练模型//KNearestModel.xml"); 


//、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、



	Mat imggray, imgblur, imgCanny, imgDil;	//定义图像变量

	cvtColor(img, imggray, COLOR_BGR2GRAY);
	//↑转化灰度图像:1转化为2,模式为3（灰度）
	//imshow("灰度", imggray);

	GaussianBlur(img, imgblur, Size(7, 7), 5, 0);
	//↑高斯模糊:size(定义内核大小7*7),输出屏幕上的位置偏移->5,0(x,y)

	Canny(imgblur, imgCanny, 30, 100);
	//↑坎尼边沿检测	  两个检测阈值(可更改,值越小边缘越多)

	//imshow("原图", img);
	//imshow("坎尼边沿检测", imgCanny);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	//↑定义可使用膨胀(扩充/侵蚀)的内核  (数越小,扩充越多)
	dilate(imgCanny, imgDil, kernel);
	//↑	边缘扩充
	//imshow("边缘扩充", imgDil);

	std::vector<Vec3f> circles;//存储每个圆的位置信息

		//霍夫圆
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 53, 18, 26);
	//	
	Mat imgCrop, hist;


	//+++++++++++++++++++++++++冒泡排序1+++++++++++++++++++++++++++++++
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
		cout << endl << "第 " << i + 1 << " 个圆：" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//绘制圆轮廓  
		//circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);

		//int R = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[2];//R
		//int G = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[1];//G
		//int B = img.at<Vec3b>(cvRound(circles[i][1]), cvRound(circles[i][0]))[0];//B
		//std::cout << "圆的半径是" << radius << std::endl;
		//std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;


		//Mat image = imread(IMG_PATH);
		//Mat dst1 = Mat::zeros(img.size(), img.type());
		//Mat mask = Mat::zeros(img.size(), CV_8U);



		Rect roi(circles[i][0] - 22, circles[i][1] - 22, 44, 44);
		//     Rect -> 定义矩形数据类型  名称(左上x坐标,左上y坐标,矩形宽，矩形高)
		Rect roi_2(circles[i][0] - 2.5, circles[i][1] - 2.5, 5, 5);

		//rectangle(img, Point(circles[i][0] - 5, circles[i][1] - 5), Point(circles[i][0] +5, circles[i][1] + 5), Scalar(255, 255, 255), FILLED);
		//				↑矩形左上角	↑右下角


		imgCrop = imggray(roi);
		//		    ↑将矩形放入图像中
		Mat imgCrop_2 = imggray(roi_2);
		//imshow("2", imgCrop_2);
		Mat imgCrop_3 = img(roi);
		//imshow("imgCrop_3", imgCrop_3);



		//###################################
		//预测分类
		////Mat src = imgCrop_3;
		//cvtColor(imgCrop_3, imgCrop_3, COLOR_BGR2GRAY);

		//threshold(imgCrop_3, imgCrop_3, 0, 255, CV_THRESH_OTSU);
		////imshow("原图像", img);
		//Mat input;
		//imgCrop_3 = imgCrop_3.reshape(1, 1);//输入图片序列化
		//input.push_back(imgCrop_3);
		//input.convertTo(input, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		//float r = knn->predict(input);   //对所有行进行预测
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


		//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&剪裁

		for (int s = 0; s < 18; s++)
		{
			imwrite("D://桌面//新建文件夹//" + to_string(s + 64) + ".png", imgCrop_3);
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

		//遍历像素++++++++++++++++++++++++++++++++++++

		//遍历图像像素方法:通过at<type>(i,j)坐标指针
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

		//灰度均值、方差---------------------------
		GetGrayAvgStdDev(imgCrop_2, a, b);

		int aa = (int)a;
		int sum_DanTongD_zhi = aa - num_s;

		if (sum_DanTongD_zhi < 0)
		{
			sum_DanTongD_zhi = sum_DanTongD_zhi * (-1);
		}
		Zhi = sum_DanTongD_zhi;



		//定义求直方图的通道数目，从0开始索引
		int channels[] = { 0 };
		//定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数
		//如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目
		const int histSize[] = { 256 };
		//每一维bin的变化范围
		float range[] = { 0,256 };

		//所有bin的变化范围，个数跟channels应该跟channels一致
		const float* ranges[] = { range };

		//定义直方图，这里求的是直方图数据
		Mat hist;
		//opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数
		calcHist(&imgCrop, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);//cv中是cvCalcHist

		//找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
		double maxValue = 0;
		//找矩阵中最大最小值及对应索引的函数
		minMaxLoc(hist, 0, &maxValue, 0, 0);
		//最大值取整
		int rows = cvRound(maxValue);
		//定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
		//因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
		Mat histImage = Mat::zeros(rows, 256, CV_8UC1);

		//直方图图像表示
		for (int i = 0; i < 256; i++)
		{
			//取每个bin的数目
			int temp = (int)(hist.at<float>(i, 0));
			//如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
			//如果图像上有该灰度值，则将该列对应个数的像素设为白色
			if (temp)
			{
				//由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
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
		//判断通道内像素点个数
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

		//由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
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
		////****运算
		//aa-sum_Dian


		//打印值
		putText(img, to_string(sum_DanTongD_zhi), Point(circles[i][0] - 10, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 0.7, Scalar(255, 0, 0), 0.5);


		//cout << "----------------------" << endl << Zhi << endl;

		//----------------------------------------------------------------------------------------------

		//int mm = 1;	
		//if ((i % 8) == 0)
		//{
		//	ofs << "-----------------" << endl;
		//	ofs << "【第 " << mm << " 列】：" << endl;
		//	ofs << "-----------------" << endl;
		//	mm++;
		//}

		ofs << "【第 " << i + 1 << " 个圆】：" << endl;

		//ofs << "圆的X是" << circles[i][0] << ",圆的Y是" << circles[i][1] << std::endl;
		ofs << "计算得出的数值：" << Zhi << endl << endl;

		//putText(img, to_string((int)circles[i][0]), Point(circles[i][0] - 5, circles[i][1] -5), FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 0, 0), 1);
		//putText(img, to_string((int)circles[i][1]), Point(circles[i][0] -5, circles[i][1]+5), FONT_HERSHEY_DUPLEX, 0.4, Scalar(255, 0, 0), 1);
		//putText(img, to_string(Zhi), Point(circles[i][0] - 18, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 0.4, Scalar(0, 0, 255), 1);


		//----------------------------------------------------------------------------------------------


		//文字
		//if (Lei == 1)
		//{
		//	putText(img, "1", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑
		//}
		//if (Lei == 2)
		//{
		//	putText(img, "2", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑
		//}
		//if (Lei == 3)
		//{
		//	putText(img, "3", Point(circles[i][0] + 1, circles[i][1] + 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 69, 255), 1);
		//	//	文字：	  ↑内容					       ↑字体（随机）		↑比例		      厚度↑
		//}
		//Lei = 0;
		//sum_Dian = 0;

		//imshow("hist"+i, hist);
		//imshow("剪裁图像+"+i, imgCrop);

	}
	imshow("【效果图】", img);
	waitKey(0);		//延迟显示

	ofs.close();
	return 0;
}

//int getHistograph( Mat grayImage,Mat imgppp)
//{
//	
//}

//计算灰度均值及方差
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



