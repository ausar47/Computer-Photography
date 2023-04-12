#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "hw3_gn.h"
#include "Solver5603.h"
#include <opencv2/opencv.hpp>

// 从文件中读取点云数据
bool readPointCloud(const std::string& filename, std::vector<std::array<double, 3>>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::array<double, 3> point;
        ss >> point[0] >> point[1] >> point[2];
        points.push_back(point);
    }

    file.close();
    return true;
}

int main() {
    std::string filename = "ellipse753.txt"; // 更改为实际数据文件名
    std::vector<std::array<double, 3>> points;

    if (!readPointCloud(filename, points)) {
        return -1;
    }

    int point_size = points.size();
    vector<vector<double>> point_array(point_size, vector<double>(3));
    for (int i = 0; i < point_size; ++i) {
        point_array[i][0] = points[i][0];
        point_array[i][1] = points[i][1];
        point_array[i][2] = points[i][2];
    }

    EllipseFunction ef(point_array, point_size);
    Solver5603 solver;

    // 设置初始值
    double X[3] = { 1.0, 1.0, 1.0 };

    // 设置优化参数
    GaussNewtonParams params;
    params.gradient_tolerance = 1e-5;
    params.residual_tolerance = 1e-5;
    params.max_iter = 1000;
    params.verbose = true;

    // 设置优化结果报告
    GaussNewtonReport* report = new GaussNewtonReport();

    // 求解最小二乘问题
    double optimal_value = solver.solve(&ef, X, params, report);

    std::cout << "Stop type: ";
    switch (report->stop_type) {
        case 0: std::cout << "梯度达到阈值";   break;
        case 1: std::cout << "余项达到阈值"; break;
        case 2: std::cout << "不收敛";  break;
        case 3: std::cout << "其它数值错误"; break;
    }
    std::cout << ", Iteration num: " << report->n_iter << std::endl;
    std::cout << "Optimal value: " << optimal_value << std::endl;
    std::cout << "AA: " << X[0] << ", BB: " << X[1] << ", CC: " << X[2] << std::endl;

    return 0;
}