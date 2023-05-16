#pragma once
#include "hw8_pa.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class Panorama5603 : public CylindricalPanorama {
public:
    // img_vec为输入的平面图像，img_out为求出的全景图，f为相机焦距
    virtual bool makePanorama(
        vector<Mat> &img_vec, Mat &img_out, double f) override {
        if (img_vec.empty()) {
            return false;
        }

        // 对输入图像进行柱面投影
        double r = 500;
        std::vector<cv::Mat> warped_imgs;
        for (const auto& img : img_vec) {
            cv::Mat warped = cylindricalProj(img, r, f);
            warped_imgs.push_back(warped);
        }

        // 迭代特征匹配并拼接
        int num = warped_imgs.size();
        int mid = num / 2;
        Mat img = warped_imgs[mid];
        for (int i = 1; i < num; i++) { // 借鉴前人经验，实测从中间开始拼接效果更好
            int index;
            if (i % 2 == 1) {
                index = mid - (i + 1) / 2;
                cout << index;
                Mat img2;
                warped_imgs[index].copyTo(img2);
                findTransform(img2, img);
                img2.copyTo(img);
            } else {
                index = mid + (i + 1) / 2;
                cout << index;
                Mat img2;
                warped_imgs[index].copyTo(img2);
                findTransform(img, img2);
            }
        }
        img.copyTo(img_out);
        return true;
    }

private:
    // 函数名：Mat cylindricalProj(const Mat& image, double r, double f)
    // 功能：将输入图像进行柱面投影变换
    // 输入参数：
    // - image：输入图像
    // - r：柱面半径
    // - f：焦距
    Mat cylindricalProj(const Mat& image, double r, double f) {
        // 获取图像的行数和列数
        int rows = image.rows;
        int cols = image.cols;

        // 计算图像中心点的坐标
        int center_x = (cols - 1) / 2;
        int center_y = (rows - 1) / 2;

        // 计算柱面投影后的图像尺寸
        int x_ = r * atan(center_x / f);
        int y_ = r * center_y / f;
        int cols_ = x_ * 2 + 1;
        int rows_ = y_ * 2 + 1;

        // 创建一个新的图像矩阵，用于存储柱面投影后的图像
        Mat M(rows_, cols_, CV_8UC3, cv::Scalar::all(1));

        // 遍历新图像的每个像素
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                // 计算原图像中对应的像素坐标
                double x = f * tan((j - x_) / r);
                double y = (i - y_) / r * sqrt(x * x + f * f);
                x = x + center_x;
                y = y + center_y;

                // 检查坐标是否越界，越界则设置为黑色
                if (x < 0 || y < 0 || x >= cols || y >= rows) {
                    M.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // 越界处理
                } else {
                    // 否则，从原图像中获取对应像素的值
                    M.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(y, x);
                }
            }
        }

        // 创建一个新的图像，用于存储扩展后的柱面投影图像
        Mat newimg(3 * rows, 4 * cols, CV_8UC3);

        // 将柱面投影图像复制到新图像的中心区域
        for (int i = 0; i < M.rows; i++) {
            for (int j = 0; j < M.cols; j++) {
                newimg.at<cv::Vec3b>(Point(j, i + rows)) = M.at<cv::Vec3b>(Point(j, i));
            }
        }

        // 返回扩展后的柱面投影图像
        return newimg;
    }

    // 函数名：void findTransform(cv::Mat& img1, const cv::Mat& img2)
    // 功能：找到两幅图像之间的单应性矩阵，并将img2变换为与img1相同的视角
    // 输入参数：
    // - img1：输入图像1
    // - img2：输入图像2
    void findTransform(cv::Mat& img1, const cv::Mat& img2) {
        Mat transform;

        // 创建特征点检测器和描述符匹配器
        cv::Ptr<cv::Feature2D> feature_detector = cv::ORB::create(5000);
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

        // 存储关键点和描述符
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;

        // 对两幅图像进行特征点检测和描述符计算
        feature_detector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
        feature_detector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

        // 匹配描述符
        std::vector<cv::DMatch> matches;
        matcher->match(descriptors1, descriptors2, matches);

        // 寻找关键点之间的最小和最大距离
        double min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {
            return m1.distance < m2.distance;
            })->distance;
        double max_dist = std::max_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {
            return m1.distance < m2.distance;
            })->distance;

        // 使用距离阈值过滤匹配结果（例如，3 * min_dist）
        std::vector<cv::DMatch> good_matches;
        for (const auto& match : matches) {
            if (match.distance <= std::max(3 * min_dist, 30.0)) {
                good_matches.push_back(match);
            }
        }

        // 从过滤后的匹配结果中提取关键点坐标
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : good_matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        // 使用RANSAC算法计算单应性矩阵
        transform = cv::findHomography(points2, points1, cv::RANSAC);

        // 使用单应性矩阵将img2变换为与img1相同的视角
        Mat wim_2;
        warpPerspective(img2, wim_2, transform, img1.size());

        // 将变换后的img2与原始img1进行融合
        for (int i = 0; i < img1.cols; i++) {
            for (int j = 0; j < img1.rows; j++) {
                Vec3b color_im1 = img1.at<Vec3b>(Point(i, j));
                Vec3b color_im2 = wim_2.at<Vec3b>(Point(i, j));

                // 如果img2的像素值较大，则使用img2的像素值替换img1的像素值
                if (norm(color_im1) < norm(color_im2)) {
                    img1.at<Vec3b>(Point(i, j)) = color_im2;
                }
            }
        }
    }
};