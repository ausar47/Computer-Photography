#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const double pi = acos(double(-1));
const int d = 5;    // neighborhood diameter

// Calculate the gaussian value
double Gaussian(double sigma, double distance) {
    return exp(-(distance * distance) / (2 * sigma * sigma)) / (2 * pi * sigma * sigma);
}

// Calculate normalize factor
double Wp(Mat image, int y, int x, double sigma_s, double sigma_r) {
    double wp = 0;
    int w = d / 2;

    for (int i = y - w; i < y + w + 1; i++) {
        for (int j = x - w; j < x + w + 1; j++) {
            double distance = sqrt(pow(x - j, 2) + pow(y - i, 2));
            double brightness_variance = abs(image.at<Vec3b>(y, x)[2] - image.at<Vec3b>(i, j)[2]);
            wp += Gaussian(sigma_s, distance) * Gaussian(sigma_r, brightness_variance);
        }
    }
    return wp;
}

// Do convolution
double Convolution(Mat image, int y, int x, double sigma_s, double sigma_r) {
    double sum = 0;
    int w = d / 2;

    for (int i = y - w; i < y + w + 1; i++) {
        for (int j = x - w; j < x + w + 1; j++) {
            double distance = sqrt(pow(x - j, 2) + pow(y - i, 2));
            double brightness_variance = abs(image.at<Vec3b>(y, x)[2] - image.at<Vec3b>(i, j)[2]);
            sum += Gaussian(sigma_s, distance) * Gaussian(sigma_r, brightness_variance) * image.at<Vec3b>(i, j)[2];
        }
    }
    return sum;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "BilateralFilter <input-image> <output-image> <sigma-s> <sigma-r>" << endl;
        return -1;
    }
    Mat image = imread(argv[1]);    // input-image
    Mat output = image.clone();     // output-image
    cvtColor(output, output, COLOR_BGR2HSV);
    double sigma_s = atoi(argv[3]); // spatial standard deviation
    double sigma_r = atoi(argv[4]); // color standard deviation

    // Check if the image is valid
    if (image.empty()) {
        cerr << "Could not open or find the image" << endl;
        return -1;
    }

    const int width = image.cols;
    const int height = image.rows;
    for (int i = d / 2; i < height - d / 2; i++) {
        for (int j = d / 2; j < width - d / 2; j++) {
            double wp = Wp(image, i, j, sigma_s, sigma_r);
            double sum = Convolution(image, i, j, sigma_s, sigma_r);
            output.at<Vec3b>(i, j)[2] = sum / wp;
        }
    }
    cvtColor(output, output, COLOR_HSV2BGR);
    imwrite(argv[2], output);
    imshow("input", image);
    imshow("output", output);

    waitKey(0);
    destroyWindow("input");
    destroyWindow("output");
    return 0;
}