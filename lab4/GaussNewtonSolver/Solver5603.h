#ifndef SOLVER_5603
#define SOLVER_5603
#include "hw3_gn.h"
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

#define POW(x) ((x)*(x))
#define POW_n3(x) (1.0/((x)*(x)*(x)))

class Solver5603 : public GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction* f, // 目标函数
		double* X,           // 输入作为初值，输出作为结果
		GaussNewtonParams param = GaussNewtonParams(), // 优化参数
		GaussNewtonReport* report = nullptr // 优化结果报告
	) override {
		int nR = f->nR(); // 获取残差向量的维度
		int nX = f->nX(); // 获取变量 X 的维度

		// 初始化残差向量 R，雅可比矩阵 J 和更新向量 deltaX
		cv::Mat R(nR, 1, CV_64F);
		cv::Mat J(nR, nX, CV_64F);
		cv::Mat deltaX(nX, 1, CV_64F);

		// 迭代求解
		for (int iter = 0; iter < param.max_iter; ++iter) {
			// 计算残差向量 R 和雅可比矩阵 J
			f->eval(R.ptr<double>(), J.ptr<double>(), X);

			// 计算残差并检查是否满足停止条件
			double residual = cv::norm(R);
			if (residual < param.residual_tolerance) {
				if (report) {
					report->stop_type = GaussNewtonReport::STOP_RESIDUAL_TOL;
					report->n_iter = iter;
				}
				break;
			}

			// 计算梯度并检查是否满足停止条件
			cv::Mat gradient = -J.t() * R;
			if (cv::norm(gradient) < param.gradient_tolerance) {
				if (report) {
					report->stop_type = GaussNewtonReport::STOP_GRAD_TOL;
					report->n_iter = iter;
				}
				break;
			}

			// 计算 Hessian 矩阵并求解线性系统 H * deltaX = gradient
			cv::Mat H = J.t() * J;
			cv::solve(H, gradient, deltaX, cv::DECOMP_CHOLESKY);

			// 更新 X
			for (int i = 0; i < nX; ++i) {
				X[i] += deltaX.at<double>(i, 0);
			}

			// 如果需要，打印每次迭代的信息
			if (param.verbose) {
				std::cout << "Iteration " << iter << ": Residual = " << residual << std::endl;
			}
		}

		// 计算最终的残差向量 R 和雅可比矩阵 J
		f->eval(R.ptr<double>(), J.ptr<double>(), X);
		// 返回最终的残差
		return cv::norm(R);
	}
};

class EllipseFunction : public ResidualFunction {
private:
    int size;
    double *x, *y, *z;

public:
    EllipseFunction(vector<vector<double>> point, int point_size) {
        size = point_size;
        x = new double[size];
        y = new double[size];
        z = new double[size];
        for (int i = 0; i < size; ++i) {
            x[i] = point[i][0];
            y[i] = point[i][1];
            z[i] = point[i][2];
        }
    }

    ~EllipseFunction() {
        delete[] x;
        delete[] y;
        delete[] z;
    }

    // 返回余项向量的维度
    virtual int nR() const override {
        return size;
    }

    // 返回变量 X 的维度
    virtual int nX() const override {
        return 3;
    }

    // 运行时读入 X 将计算得到的余项和Jacobian写入 R 和 J
    virtual void eval(double *R, double *J, double *X) override {
        for (int i = 0; i < size; ++i) {
            double A = X[0];
            double B = X[1];
            double C = X[2];

            R[i] = 1 - (x[i] * x[i]) / (A * A) - (y[i] * y[i]) / (B * B) - (z[i] * z[i]) / (C * C);

            J[i * 3 + 0] = 2 * x[i] * x[i] / (A * A * A);
            J[i * 3 + 1] = 2 * y[i] * y[i] / (B * B * B);
            J[i * 3 + 2] = 2 * z[i] * z[i] / (C * C * C);
        }
    }
};

#endif /* SOLVER_5603 */