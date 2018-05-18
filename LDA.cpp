#include "Eigen/Eigen"
#include <opencv2\opencv.hpp>
#include <opencv2\core\eigen.hpp>
#include <opencv2\core\core.hpp>

#include "iostream"

using namespace cv;
using namespace std;
#define M 7
#define Mp 4
#define Mn 3
#define N 2
static int way;

Mat getVecofMaxVal(Mat eigenVal, Mat eigenVec, int k);

int main()
{
	double sampledata[M][N] = { { 2.95,6.63 },{ 2.53,7.79 },{ 3.57,5.65 },
	   { 3.16,5.47 },{ 2.58,4.46 },{ 2.16,6.22 },{ 3.27,3.52 } };
	Mat dataMat = Mat(7, 2, CV_64FC1, sampledata);
	//cout << dataMat<<endl<<dataMat.rows<<" "<<dataMat.cols;

	int dataLable[M] = { 0,0,0,0,1,1,1 };
	Mat lableMat = Mat(7, 1, CV_32SC1, dataLable);
	cout << "哪种方法：";
	cin >> way;

	if (way == 0) 
	{
		LDA lda = LDA();
		lda.compute(dataMat, lableMat);

		Mat w = lda.eigenvectors();
		cout <<"投影矩阵" << w << endl;

		Mat dataPro = lda.project(dataMat);
		cout << "投影值" << dataPro << endl;
	}
	
	else
	{
		//1.求均值和协方差,reduce是个好东西(可求每一行/列 和 均值 最大 最小) ，rowrange(a,b)取第a行到b-1行

		Mat data_pmean, data_nmean;
		reduce(dataMat.rowRange(0, 4), data_pmean, 0, REDUCE_AVG);
		reduce(dataMat.rowRange(4, 7), data_nmean, 0, REDUCE_AVG);
		//cout << data_pmean << endl<<data_nmean<<endl;

		Mat data_psigma, data_nsigma;
		for (int i = 0; i < Mp; i++)
		{
			Mat A = dataMat.row(i) - data_pmean;
			data_psigma = data_psigma + A.t()*A;
		}
		for (int i = Mp; i < M; i++)
		{
			Mat  A = dataMat.row(i) - data_nmean;
			data_nsigma = data_nsigma + A.t()*A;
		}

		//2.计算类内散度矩阵和类间散度矩阵
		Mat Sw = data_psigma + data_nsigma; //类内散度矩阵
		Mat Sb;		//类间散度矩阵
		Mat B = data_pmean - data_nmean;
		Sb = B.t()*B;
		cout << "类内散度矩阵"<<Sw << endl <<"类间散度矩阵"<< Sb << endl;

		//3.计算投影矩阵w  ///对于二分类 可直接求 w = Sw.inv()*B.t();
		// opencv中特征向量是row ，eigen中特征向量是col(而且是复数)
		Mat w,eigenVal,eigenVec;
		Eigen::MatrixXd matrix;
		Eigen::MatrixXcd val, vec;
		Eigen::MatrixXd valReal,vecReal;

		cv2eigen(Sw.inv()*Sb, matrix);

		Eigen::EigenSolver<Eigen::MatrixXd> es(matrix);
		cout << "特征值" << es.eigenvalues() << endl;
		cout << "eigen特征向量" << es.eigenvectors() << endl;

		valReal = es.eigenvalues().real();
		vecReal = es.eigenvectors().real().transpose();

		eigen2cv(valReal, eigenVal);
		eigen2cv(vecReal, eigenVec);


		cout << "特征值" << eigenVal << endl;
		cout << "opencv特征向量" << eigenVec << endl;
		w = getVecofMaxVal(eigenVal, eigenVec, 1);
		
		cout << "投影矩阵" << w << endl;

		Mat dataPro = dataMat * w.t();
		cout << "投影值" << dataPro<<endl;
		cout << endl;
	}
	
	return 0;
}

//求最大特征值对应的特征向量
Mat getVecofMaxVal(Mat eigenVal, Mat eigenVec, int k)
{
	Mat VecofMaxVal = Mat(k, N, CV_64FC1);
	Mat c;
	double d;
	for (int i = 0; i<N - 1; i++)
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
	VecofMaxVal = eigenVec.rowRange(0, k);
	return VecofMaxVal;
}