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

	//图片读取路径
	string path = "D://桌面//YY_2.jpg";
	Mat img = imread(path);
	//imshow("原图", img);

	if (img.empty())
	{
		cerr << "未找到文件！！！" << endl;
		return -1;
	}

	//============================== = 读取训练数据============================== =

	const int classsum = 3;//图片共有6类
	const int imagesSum = 82;//每类有82张图片			   
	const int imageRows = 44;//图片尺寸
	const int imageCols = 44;
	//训练数据，每一行一个训练图片
	Mat trainingData;
	//训练样本标签
	Mat labels;
	//最终的训练样本标签
	Mat clas;
	//最终的训练数据
	Mat traindata;

	//从指定文件夹下提取图片//
	for (int p = 0; p < classsum; p++)
	{
		//训练集读取路径
		oss << "D://桌面//训练//";
		num += 1;//num从0到2
		int label = num;
		oss << num << "//*.png";	//图片名字后缀，oss可以结合数字与字符串
		string pattern = oss.str();	//oss.str()输出oss字符串，并且赋给pattern
		oss.str("");				//每次循环后把oss字符串清空
		vector<Mat> input_images;
		vector<String> input_images_name;
		glob(pattern, input_images_name, false);
		//为false时，仅仅遍历指定文件夹内符合模式的文件，当为true时，会同时遍历指定文件夹的子文件夹
		//此时input_images_name存放符合条件的图片地址

		int all_num = input_images_name.size();//文件下总共有几个图片
		cout << num << ":总共有" << all_num << "个图片待测试" << endl;
		cout << "-------------------------------" << endl;

		for (int i = 0; i < imagesSum; i++)
		{
			cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);
			threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);
			input_images.push_back(yangben_thresh);
			//循环读取每张图片并且依次放在vector<Mat> input_images内
			dealimage = input_images[i];

			//利用reshape()函数完成特征提取,
			//reshape(1, 1)的结果就是原图像对应的矩阵将被拉伸成一个一行的向量，作为特征向量。
			dealimage = dealimage.reshape(1, 1);//图片序列化
			trainingData.push_back(dealimage);//序列化后的图片依次存入
			labels.push_back(label);//把每个图片对应的标签依次存入
		}
	}

	//图片数据和标签转变下
	Mat(trainingData).copyTo(traindata);//复制
	traindata.convertTo(traindata, CV_32FC1);//更改图片数据的类型，必要，不然会出错
	Mat(labels).copyTo(clas);//复制


	//============================== = 创建KNN模型============================== =

	Ptr<KNearest>knn = KNearest::create();
	knn->setDefaultK(10);//k个最近领
	knn->setIsClassifier(true);//true为分类，false为回归
	//训练数据和标签的结合
	Ptr<TrainData>trainData = TrainData::create(traindata, ROW_SAMPLE, clas);
	//训练
	knn->train(trainData);

	//model->save("D://训练模型//KNearestModel.xml"); 


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

	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	//↑定义可使用膨胀(扩充/侵蚀)的内核  (数越小,扩充越多)
	dilate(imgCanny, imgDil, kernel);
	//↑	边缘扩充
	namedWindow("边缘扩充", WINDOW_FREERATIO);
	imshow("边缘扩充", imgDil);

	std::vector<Vec3f> circles;//存储每个圆的位置信息

		//霍夫圆
	HoughCircles(imgCanny, circles, CV_HOUGH_GRADIENT, 1.8, 10, 20, 60, 80, 120);
	//	
	Mat imgCrop, hist;


	//+++++++++++++++++++++++++冒泡排序1+++++++++++++++++++++++++++++++
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

	//遍历所有圆：
	for (size_t i = 0; i < circles.size(); i++)
	{
		cout << endl << "第 " << i + 1 << " 个圆：" << endl << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//x,y
		int radius = cvRound(circles[i][2]);//r

		//绘制圆轮廓  
		//circle(img, center, radius + 1, Scalar(155, 50, 255), 3, 8, 0);

		std::cout << "圆的半径是" << radius << std::endl;
		std::cout << "圆的X是" << circles[i][0] << "圆的Y是" << circles[i][1] << std::endl;
		cout << "----------------------------------------" << endl;

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

	//==============================预测分类============================== =

		cvtColor(imgCrop_3, imgCrop_3, COLOR_BGR2GRAY);
		threshold(imgCrop_3, imgCrop_3, 0, 255, CV_THRESH_OTSU);
		//imshow("原图像", img);
		Mat input;
		imgCrop_3 = imgCrop_3.reshape(1, 1);//输入图片序列化
		input.push_back(imgCrop_3);
		input.convertTo(input, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = knn->predict(input);   //对所有行进行预测
		//cout << r << endl;

		int rr = (int)r;

		//分类  将类别输出到图片
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

		//&&&&&&&&&&&&&&&&&&-----采集样本-----&&&&&&&&&&&&&&&&&		
		//for (int s = 0; s < 6; s++)
		//{
		//	
		//}
		//imwrite("D://桌面//新建文件夹//" + to_string(i-16) + ".png", imgCrop_3);
		//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		//----------------------------------------------------------------------------------------------

	}
	imshow("【效果图】", img);
	waitKey(0);

	return 0;
}

