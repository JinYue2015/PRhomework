#include <opencv2/opencv.hpp>
#include "iostream"
#include <bitset>

using namespace cv;
using namespace std;

void LBP(Mat src, Mat &dst, int points, int radius);

int main()
{
	Mat srcImg, dstImg;
	srcImg = imread("D:/Document/����/att_faces/s1/1.pgm", 1);

	LBP(srcImg, dstImg, 8, 2);
	imshow("",dstImg);
	waitKey();
	
	return 0;
}

void LBP(Mat src, Mat &dst, int points, int radius)
{
	Mat gray_img;
	if (src.channels() == 3)
	{
		cvtColor(src, gray_img, CV_BGR2GRAY);
	}
	else
	{
		gray_img = src;
	}

	dst = Mat::zeros(gray_img.rows - 2 * radius, gray_img.cols - 2 * radius, CV_8UC1);

	//���������λ�ã����ټ�����
	for (auto k = 0; k < points; k++)
	{
		//������λ�ã�˫���Բ�ֵ
		float rx = radius * cos(CV_PI * 2 * k / points);
		float ry = -radius * sin(CV_PI * 2 * k / points);

		int rx1 = floor(rx);
		int rx2 = ceil(rx);
		int ry1 = floor(ry);
		int ry2 = ceil(ry);

		float u = rx - rx1;
		float v = ry - ry1;

		float w1 = (1 - u)*(1 - v);
		float w2 = (1 - u)*v;
		float w3 = u * (1 - v);
		float w4 = u * v;

		//�ԻҶ�ͼÿ�������LBP����
		for (int i = radius; i< gray_img.rows - radius; i++)
		{ for (int j = radius; j < gray_img.cols - radius; j++)
			{
				uchar center = gray_img.at<uchar>(i, j);
				uchar neighbor_value = gray_img.at<uchar>(i + rx1, j + ry1)*w1 
					+ gray_img.at<uchar>(i + rx1, j + ry2)*w2
					+ gray_img.at<uchar>(i + rx2, j + ry1)*w3
					+ gray_img.at<uchar>(i + rx2, j + ry2)*w4;
				dst.at<uchar>(i - radius, j - radius) |= (neighbor_value > center) << (points - k - 1);
			}
		}	
	}

	//��ת����ģʽ��ȡ��ת����Сֵ
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			uchar val = dst.at<uchar>(i, j);
			uchar minval = val;

			for (auto k = 1; k < points; k++)
			{
				uchar tempval = (val >> (points - k)) | (val << k);
				if (tempval < minval)
				{
					minval = tempval;
				}
			}
			dst.at<uchar>(i, j) = minval;
		}
	}

	//�ȼ�ģʽ�����������������ת����ģʽ������еȼ۱���
	uchar pattern = 1;
	uchar code_table[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		int count = 0;
		bitset<8> binaryCode = i;

		//����
		for (int m = 0; m < 8; m++)
		{
			if (binaryCode[m] != binaryCode[(m + 1) % 8]) count++;
		}

		if (count < 3)
		{
			code_table[i] = pattern;
			pattern++;
		}
		else
			code_table[i] = 0;
	}
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
		{ 
			dst.at<uchar>(i, j) = code_table[dst.at<uchar>(i, j)];
		}
			

}