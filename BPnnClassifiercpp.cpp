#include <Eigen\Eigen>
#include "iostream"

using namespace std;
using namespace Eigen;

static int itr = 50000;
static float lr = 0.05;

MatrixXf sigmoid(MatrixXf m);
MatrixXf dsigmoid(MatrixXf m);
VectorXf softmax(VectorXf z);

int main()
{
	double x_data[5][3] = { { 1,2,3 } ,{ 2,3,4 } ,{ 5,6,7 },{ 6,7,8 },{ 7,8,9 } };
	double y_data[5][2] = { { 1,0 } ,{ 1,0 } ,{ 1,0 },{ 0,1 },{ 0,1 } };

	MatrixXf xm(5, 3);   xm << 1, 2, 3, 2, 3, 4, 5, 6, 7, 6, 7, 8, 7, 8, 9;
	MatrixXf ym(5, 2);   ym << 1, 0, 1, 0, 1, 0, 0, 1, 0,1;
	MatrixXf xmm = xm.transpose();
	MatrixXf ymm(2, 5);ymm = ym.transpose();
	Vector3f x, x1, z1, phy1;
	Vector2f y, x2, z2;
	Vector2f phy2;

	MatrixXf w1 = MatrixXf::Random(3, 3);
	Vector3f b1(0, 0, 0);

	MatrixXf w2 = MatrixXf::Random(2, 3);
	Vector2f b2(0, 0);

	Matrix3Xf dw1;
	Matrix2Xf dw2;
	Vector3f db1;
	Vector2f db2;

	MatrixXf dL(1, 2);
	MatrixXf mat1(2, 3), mat2(1, 2);

	float loss;

	for (int i = 0; i < itr; i++)
	{
		x << x_data[i % 5][0], x_data[i % 5][1], x_data[i % 5][2];
		y << y_data[i % 5][0], y_data[i % 5][1];

		z1 = w1 * x + b1;
		x1 = sigmoid(w1 * x + b1);
		z2 = w2 * x1 + b2;
		//x2 = sigmoid(w2 * x1 + b2);

		phy1 = dsigmoid(z1);
		//phy2 = dsigmoid(z2);

		//dL << -2 * (y(0, 0) - x2(0, 0)), -2 * (y(1, 0) - x2(1, 0));
		//mat2 = phy2.transpose().cwiseProduct(dL);
		
		phy2 = softmax(z2);
		//cout <<y <<endl<< phy2 << endl << endl;
		mat2 << y(0)*(phy2(0)-1 )+y(1)*phy2(0), y(0)*phy2(1)+ y(1)*( phy2(1)-1);

		mat1.row(0) = phy1.transpose().cwiseProduct(w2.row(0));
		mat1.row(1) = phy1.transpose().cwiseProduct(w2.row(1));

		dw1 = (x * mat2 * mat1).transpose();
		db1 = (mat2 * mat1).transpose();
		dw2 = (x1 * mat2).transpose();
		db2 = mat2.transpose();

		w1 -= lr * dw1;
		b1 -= lr * db1;
		w2 -= lr * dw2;
		b2 -= lr * db2;

		if (i % 100 == 0)
		{
			//cout << dw1 << endl<<endl;
			MatrixXf y2(2, 5);
			for (int j = 0; j < 5; j++)
			{
				y2.col(j) = softmax(w2 * sigmoid(w1 *xmm.col(j) + b1) + b2);
			}
			//cout << y2 << endl;
			y2 = y2.array().log();
			loss =-ymm.cwiseProduct(y2).sum();
			//loss = (ymm - y2).array().square().sum();
			cout << "loss:" << loss << endl;
		}
	}
	cout << "w1:" << endl << w1 << endl
		<< " b1:" << endl << b1 << endl
		<< " w2:" << endl << w2 << endl
		<< " b2:" << endl << b2 << endl<<endl;

	Vector3f x_test(6, 7, 8);
	VectorXf y_test;
	y_test = softmax(w2*sigmoid(w1*x_test + b1) + b2);
	cout << y_test;


}

MatrixXf sigmoid(MatrixXf m)
{
	MatrixXf result;
	float s = 1.0;
	m = -m;
	m = m.array().exp();	//逐元素计算指数函数
	m.array() += s;			//逐元素加法（s为标量）

	result = m.cwiseInverse();////逐元素取倒数
	return result;
}

//r=1/(2+ e-x + ex)
MatrixXf dsigmoid(MatrixXf m)
{
	MatrixXf result;
	m = m.array().exp() + (-m).array().exp();
	m.array() += 2;
	result = m.cwiseInverse();
	return result;
}

VectorXf softmax(VectorXf z)
{
	VectorXf result;
	VectorXf temp;
	double fsum;
	result.resize(z.size());
	temp = z.array().exp();
	fsum = temp.sum();
	for (int i = 0; i < z.rows(); i++)
	{
		result(i) = exp(z(i)) / fsum;
		if (result(i) < 0.000001) result(i) = 0;
	}
		

	return result;

}