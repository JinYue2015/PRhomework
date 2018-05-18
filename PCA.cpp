#include <opencv2/opencv.hpp>
#include "iostream"

using namespace cv;
using namespace std;
#define M 7
#define MM 10
#define N 3
#define N_2 2 

Mat getVecofMaxVal(Mat eigenval, Mat eigenVec, int k);

int main()
{
	double sampledata[M][N]= { { 2.95,6.63 ,2},{ 2.53,7.79 ,3},{ 3.57,5.65 ,5},
	{ 3.16,5.47 ,0},{ 2.58,4.46,4},{ 2.16,6.22 ,1},{ 3.27,3.52 ,7} };
	Mat dataMat = Mat(M, N, CV_64FC1, sampledata);

	//1.中心化
	Mat data_mean;
	reduce(dataMat, data_mean, 0, REDUCE_AVG);
	cout << data_mean<<endl;
	for (int i = 0; i < M; i++)
	{
		Mat A = dataMat.row(i) - data_mean;
		A.copyTo(dataMat.row(i));
	}
	cout << dataMat<<endl;
	
	//2.求协方差矩阵
	Mat sigma = dataMat.t()*dataMat/(M-1);

	//3.求特征值特征向量 cv::eigen只能求对称矩阵的特征值特征向量
	Mat eigenVal,eigenVec;
	eigen(sigma, eigenVal, eigenVec);
	cout << sigma << endl;
	cout << eigenVal << endl;
	cout << eigenVec << endl;

	//4.求投影阵
	Mat w2;
	w2 = getVecofMaxVal(eigenVal, eigenVec, N_2);
	cout << w2 << endl;

	return 0;
}

//求最大特征值对应的特征向量
Mat getVecofMaxVal(Mat eigenVal, Mat eigenVec, int k)
{
	Mat VecofMaxVal = Mat(k, N, CV_64FC1);
	Mat c;
	double d;
	for (int i=0;i<N-1;i++)
		for (int j = i + 1; j < N; j++)
		{
			if (eigenVal.at<double>(j, 0) > eigenVal.at<double>(i, 0))
			{
				d = eigenVal.at<double>(i, 0);
				eigenVal.at<double>(i, 0) = eigenVal.at<double>(j, 0);
				eigenVal.at<double>(j, 0) = d;

				eigenVec.row(i).copyTo(c);
				eigenVec.row(j).copyTo(eigenVec.row(i));
				c.copyTo(eigenVec.row(j));
			}
		}
	VecofMaxVal = eigenVec.rowRange(0, k );
	return VecofMaxVal;
}