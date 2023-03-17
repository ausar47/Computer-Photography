#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const double pi = acos(double(-1));

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cerr << "GaussianFilter <input-image> <output-image> <sigma>" << endl;
        return -1;
    }
    Mat image = imread(argv[1]);	// input-image
    Mat output = imread(argv[2]);	// output-image
    double sigma = atof(argv[3]);	/*
                                    Gaussian kernel parameter,
                                    This parameter controls how spread out the values are around the origin.
                                    A larger sigma means a wider and flatter curve,
                                    while a smaller sigma means a narrower and sharper curve.
                                    */

                                    // Check if the image is valid
    if (image.empty()) {
        cerr << "Could not open or find the image" << endl;
        return -1;
    }

    int size = 2 * 5 * (int)sigma + 1;
    double sum = 0;
    int center = size / 2;
    Mat kernel(size, size, CV_64F);  // Gaussian kernel, a single-channel floating point matrix
    // Calculate the value of each element of the matrix using the formula: k[i][j] = exp(-((i-m)^2 + (j-m)^2) / (2 * sigma^2))
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel.at<double>(i, j) = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
            sum += kernel.at<double>(i, j);
        }
    }

    /*
    Normalize the matrix by dividing each element by the sum of all elements.
    This ensures that the matrix has a total weight of 1,
    which preserves the brightness of the image or signal after applying the kernel.
    */
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel.at<double>(i, j) /= sum;
        }
    }

    filter2D(image, output, image.depth(), kernel); // do the convolution

    imwrite(argv[2], output);
    imshow("input", image);
    imshow("output", output);

    waitKey(0);
    destroyWindow("input");
    destroyWindow("output");
    return 0;
}