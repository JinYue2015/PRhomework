#include <opencv2\opencv.hpp>
#include "iostream"

using namespace cv;
using namespace std;

#define pi 3.1415926

Mat GaussianKenerl(int num);
void GaborKernel(Mat &Gkenerl, Mat &GaborReal, Mat &GaborImag,float u, float v);

int main()
{
	Mat srcImg, dstImg_r,dstImg_i;
	Mat gKenerl,gaborReal,gaborImag;
	
	srcImg = imread("D:/Document/³ÌÐò/att_faces/s2/1.pgm", 0);
	imshow("src", srcImg);
	gKenerl = GaussianKenerl(11);

	gaborReal.create(gKenerl.size(), CV_32FC1);
	gaborImag.create(gKenerl.size(), CV_32FC1);


	GaborKernel(gKenerl, gaborReal, gaborImag, 0.0125, 0.0125);
	/*cout << gKenerl<<endl;
	cout << gaborReal << endl;
	cout << gaborImag << endl;*/
	
	//cout << srcImg;
	filter2D(srcImg, dstImg_r, CV_8UC1, gaborReal);
	filter2D(srcImg, dstImg_i, CV_32FC1, gaborImag);

	//imshow("Gaussian", gKenerl);
	//imshow("Real", gaborReal);
	//imshow("imag", gaborImag);
	imshow("realresult", dstImg_r);
	imshow("imagresult", dstImg_i);
	waitKey();
}

Mat GaussianKenerl(int num)
{
	Mat Gkenerl = Mat::zeros(num, num, CV_32F);
	Point middle(int((num - 1) / 2), int((num - 1) / 2));
	float xr, yr;
	float theta=90.0/180.0*pi;
	float ox=5, oy=10;
	if (num % 2 == 1)
	{
		for (int i = 0; i<Gkenerl.rows; i++)
			for (int j = 0; j < Gkenerl.cols; j++)
			{
				xr = float(i - middle.x)*cos(theta) + float(j - middle.y)*sin(theta);
				yr = -float(i - middle.x)*sin(theta) + float(j - middle.y)*cos(theta);
				Gkenerl.at<float>(i, j) = exp(-float((xr*xr/ox + yr*yr/oy)) )/(pi*2);
			}
	}
	return Gkenerl;
}
void GaborKernel(Mat &Gkenerl,Mat &GaborReal, Mat &GaborImag, float u, float v)
{
	Mat w;
	Gkenerl.copyTo(w);
	for (int i=0 ; i<w.rows; i++)
		for (int j=0; j<w.cols; j++)
		{
			GaborReal.at<float>(i, j) = w.at<float>(i, j) * cos((2 * pi *(u*(i - (w.rows - 1)*0.5) + v * (j - (w.cols - 1)*0.5))) / 180.0*pi);
			GaborImag.at<float>(i, j) = w.at<float>(i, j) * sin((2 * pi *(u*(i - (w.rows - 1)*0.5) + v * (j - (w.cols - 1)*0.5))) / 180.0*pi);

		}
}
