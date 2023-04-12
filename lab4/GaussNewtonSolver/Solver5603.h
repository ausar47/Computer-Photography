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
		ResidualFunction* f, // Ŀ�꺯��
		double* X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport* report = nullptr // �Ż��������
	) override {
		int nR = f->nR(); // ��ȡ�в�������ά��
		int nX = f->nX(); // ��ȡ���� X ��ά��

		// ��ʼ���в����� R���ſɱȾ��� J �͸������� deltaX
		cv::Mat R(nR, 1, CV_64F);
		cv::Mat J(nR, nX, CV_64F);
		cv::Mat deltaX(nX, 1, CV_64F);

		// �������
		for (int iter = 0; iter < param.max_iter; ++iter) {
			// ����в����� R ���ſɱȾ��� J
			f->eval(R.ptr<double>(), J.ptr<double>(), X);

			// ����в����Ƿ�����ֹͣ����
			double residual = cv::norm(R);
			if (residual < param.residual_tolerance) {
				if (report) {
					report->stop_type = GaussNewtonReport::STOP_RESIDUAL_TOL;
					report->n_iter = iter;
				}
				break;
			}

			// �����ݶȲ�����Ƿ�����ֹͣ����
			cv::Mat gradient = -J.t() * R;
			if (cv::norm(gradient) < param.gradient_tolerance) {
				if (report) {
					report->stop_type = GaussNewtonReport::STOP_GRAD_TOL;
					report->n_iter = iter;
				}
				break;
			}

			// ���� Hessian �����������ϵͳ H * deltaX = gradient
			cv::Mat H = J.t() * J;
			cv::solve(H, gradient, deltaX, cv::DECOMP_CHOLESKY);

			// ���� X
			for (int i = 0; i < nX; ++i) {
				X[i] += deltaX.at<double>(i, 0);
			}

			// �����Ҫ����ӡÿ�ε�������Ϣ
			if (param.verbose) {
				std::cout << "Iteration " << iter << ": Residual = " << residual << std::endl;
			}
		}

		// �������յĲв����� R ���ſɱȾ��� J
		f->eval(R.ptr<double>(), J.ptr<double>(), X);
		// �������յĲв�
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

    // ��������������ά��
    virtual int nR() const override {
        return size;
    }

    // ���ر��� X ��ά��
    virtual int nX() const override {
        return 3;
    }

    // ����ʱ���� X ������õ��������Jacobianд�� R �� J
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