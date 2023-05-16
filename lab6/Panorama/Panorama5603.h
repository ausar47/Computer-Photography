#pragma once
#include "hw8_pa.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class Panorama5603 : public CylindricalPanorama {
public:
    // img_vecΪ�����ƽ��ͼ��img_outΪ�����ȫ��ͼ��fΪ�������
    virtual bool makePanorama(
        vector<Mat> &img_vec, Mat &img_out, double f) override {
        if (img_vec.empty()) {
            return false;
        }

        // ������ͼ���������ͶӰ
        double r = 500;
        std::vector<cv::Mat> warped_imgs;
        for (const auto& img : img_vec) {
            cv::Mat warped = cylindricalProj(img, r, f);
            warped_imgs.push_back(warped);
        }

        // ��������ƥ�䲢ƴ��
        int num = warped_imgs.size();
        int mid = num / 2;
        Mat img = warped_imgs[mid];
        for (int i = 1; i < num; i++) { // ���ǰ�˾��飬ʵ����м俪ʼƴ��Ч������
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
    // ��������Mat cylindricalProj(const Mat& image, double r, double f)
    // ���ܣ�������ͼ���������ͶӰ�任
    // ���������
    // - image������ͼ��
    // - r������뾶
    // - f������
    Mat cylindricalProj(const Mat& image, double r, double f) {
        // ��ȡͼ�������������
        int rows = image.rows;
        int cols = image.cols;

        // ����ͼ�����ĵ������
        int center_x = (cols - 1) / 2;
        int center_y = (rows - 1) / 2;

        // ��������ͶӰ���ͼ��ߴ�
        int x_ = r * atan(center_x / f);
        int y_ = r * center_y / f;
        int cols_ = x_ * 2 + 1;
        int rows_ = y_ * 2 + 1;

        // ����һ���µ�ͼ��������ڴ洢����ͶӰ���ͼ��
        Mat M(rows_, cols_, CV_8UC3, cv::Scalar::all(1));

        // ������ͼ���ÿ������
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                // ����ԭͼ���ж�Ӧ����������
                double x = f * tan((j - x_) / r);
                double y = (i - y_) / r * sqrt(x * x + f * f);
                x = x + center_x;
                y = y + center_y;

                // ��������Ƿ�Խ�磬Խ��������Ϊ��ɫ
                if (x < 0 || y < 0 || x >= cols || y >= rows) {
                    M.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // Խ�紦��
                } else {
                    // ���򣬴�ԭͼ���л�ȡ��Ӧ���ص�ֵ
                    M.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(y, x);
                }
            }
        }

        // ����һ���µ�ͼ�����ڴ洢��չ�������ͶӰͼ��
        Mat newimg(3 * rows, 4 * cols, CV_8UC3);

        // ������ͶӰͼ���Ƶ���ͼ�����������
        for (int i = 0; i < M.rows; i++) {
            for (int j = 0; j < M.cols; j++) {
                newimg.at<cv::Vec3b>(Point(j, i + rows)) = M.at<cv::Vec3b>(Point(j, i));
            }
        }

        // ������չ�������ͶӰͼ��
        return newimg;
    }

    // ��������void findTransform(cv::Mat& img1, const cv::Mat& img2)
    // ���ܣ��ҵ�����ͼ��֮��ĵ�Ӧ�Ծ��󣬲���img2�任Ϊ��img1��ͬ���ӽ�
    // ���������
    // - img1������ͼ��1
    // - img2������ͼ��2
    void findTransform(cv::Mat& img1, const cv::Mat& img2) {
        Mat transform;

        // ����������������������ƥ����
        cv::Ptr<cv::Feature2D> feature_detector = cv::ORB::create(5000);
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

        // �洢�ؼ����������
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;

        // ������ͼ������������������������
        feature_detector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
        feature_detector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

        // ƥ��������
        std::vector<cv::DMatch> matches;
        matcher->match(descriptors1, descriptors2, matches);

        // Ѱ�ҹؼ���֮�����С��������
        double min_dist = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {
            return m1.distance < m2.distance;
            })->distance;
        double max_dist = std::max_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {
            return m1.distance < m2.distance;
            })->distance;

        // ʹ�þ�����ֵ����ƥ���������磬3 * min_dist��
        std::vector<cv::DMatch> good_matches;
        for (const auto& match : matches) {
            if (match.distance <= std::max(3 * min_dist, 30.0)) {
                good_matches.push_back(match);
            }
        }

        // �ӹ��˺��ƥ��������ȡ�ؼ�������
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : good_matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        // ʹ��RANSAC�㷨���㵥Ӧ�Ծ���
        transform = cv::findHomography(points2, points1, cv::RANSAC);

        // ʹ�õ�Ӧ�Ծ���img2�任Ϊ��img1��ͬ���ӽ�
        Mat wim_2;
        warpPerspective(img2, wim_2, transform, img1.size());

        // ���任���img2��ԭʼimg1�����ں�
        for (int i = 0; i < img1.cols; i++) {
            for (int j = 0; j < img1.rows; j++) {
                Vec3b color_im1 = img1.at<Vec3b>(Point(i, j));
                Vec3b color_im2 = wim_2.at<Vec3b>(Point(i, j));

                // ���img2������ֵ�ϴ���ʹ��img2������ֵ�滻img1������ֵ
                if (norm(color_im1) < norm(color_im2)) {
                    img1.at<Vec3b>(Point(i, j)) = color_im2;
                }
            }
        }
    }
};