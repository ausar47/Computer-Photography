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
    vector<Mat> imgVec; // ���ڴ洢ͼƬ����
    Mat imgOut; // ���ͼ��
    for (int i = 0; i < data2.size(); ++i) { // ��ÿ��ͼƬ�ļ����в���
        Mat src = imread(data2[i]); // ��ȡͼƬ����
        if (src.empty()) { // �����ȡʧ��
            cout << "��ȡͼƬʧ�ܣ�" << endl;
            return -1;
        }
        imgVec.push_back(src); // ��ͼƬ���ݴ���Mat����
    }
    cout << "�ɹ���ȡ" << imgVec.size() << "��ͼƬ��" << endl;
    ifstream inFile("./panorama-data2/K.txt"); // ���ļ�
    if (!inFile) { // �����ʧ��
        cout << "�޷����ļ���" << endl;
        return 1;
    }
    double f; // ����
    inFile >> f; // ���ļ��ж�ȡһ��������
    cout << f << endl; // �����ȡ�ĸ�����
    inFile.close(); // �ر��ļ�
    Panorama5603 panorama;
    panorama.makePanorama(imgVec, imgOut, f);
    imshow("result", imgOut);
    imwrite("result.JPG", imgOut);
    waitKey(0);
    return 0;
}