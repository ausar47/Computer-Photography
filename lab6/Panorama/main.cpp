#include "Panorama5603.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<string> data1 = {
    "./panorama-data1/DSC01538.jpg",
    "./panorama-data1/DSC01539.jpg",
    "./panorama-data1/DSC01540.jpg",
    "./panorama-data1/DSC01541.jpg",
    "./panorama-data1/DSC01542.jpg",
    "./panorama-data1/DSC01543.jpg",
    "./panorama-data1/DSC01544.jpg",
    "./panorama-data1/DSC01545.jpg",
    "./panorama-data1/DSC01546.jpg",
    "./panorama-data1/DSC01547.jpg",
    "./panorama-data1/DSC01548.jpg",
    "./panorama-data1/DSC01549.jpg",
};

vector<string> data2 = {
    "./panorama-data2/DSC01599.jpg",
    "./panorama-data2/DSC01600.jpg",
    "./panorama-data2/DSC01601.jpg",
    "./panorama-data2/DSC01602.jpg",
    "./panorama-data2/DSC01603.jpg",
    "./panorama-data2/DSC01604.jpg",
    "./panorama-data2/DSC01605.jpg",
    "./panorama-data2/DSC01606.jpg",
    "./panorama-data2/DSC01607.jpg",
    "./panorama-data2/DSC01608.jpg",
    "./panorama-data2/DSC01609.jpg",
    "./panorama-data2/DSC01610.jpg",
    "./panorama-data2/DSC01611.jpg",
    "./panorama-data2/DSC01612.jpg",
    "./panorama-data2/DSC01613.jpg",
    "./panorama-data2/DSC01614.jpg",
    "./panorama-data2/DSC01615.jpg",
    "./panorama-data2/DSC01616.jpg",
    "./panorama-data2/DSC01617.jpg",
    "./panorama-data2/DSC01618.jpg"
};

int main() {
    vector<Mat> imgVec; // 用于存储图片数据
    Mat imgOut; // 输出图像
    for (int i = 0; i < data2.size(); ++i) { // 对每个图片文件进行操作
        Mat src = imread(data2[i]); // 读取图片数据
        if (src.empty()) { // 如果读取失败
            cout << "读取图片失败！" << endl;
            return -1;
        }
        imgVec.push_back(src); // 将图片数据存入Mat数组
    }
    cout << "成功读取" << imgVec.size() << "张图片！" << endl;
    ifstream inFile("./panorama-data2/K.txt"); // 打开文件
    if (!inFile) { // 如果打开失败
        cout << "无法打开文件！" << endl;
        return 1;
    }
    double f; // 焦距
    inFile >> f; // 从文件中读取一个浮点数
    cout << f << endl; // 输出读取的浮点数
    inFile.close(); // 关闭文件
    Panorama5603 panorama;
    panorama.makePanorama(imgVec, imgOut, f);
    imshow("result", imgOut);
    imwrite("result.JPG", imgOut);
    waitKey(0);
    return 0;
}