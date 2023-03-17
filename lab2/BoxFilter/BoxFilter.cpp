#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "BoxFilter <input-image> <output-image> <w> <h>" << endl;
        return -1;
    }
    Mat image = imread(argv[1]);    // input-image
    Mat output = imread(argv[2]);   // output-image
    int width = atoi(argv[3]);      // distance from anchor point to left/right border
    int height = atoi(argv[4]);     // distance from anchor point to top/bottom border

    // Check if the image is valid
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat kernel(2 * width + 1, 2 * height + 1, CV_64F);  // convolution kernel, a single-channel floating point matrix
    // compute kernel, K(u, v) = 1 / |N| if (u, v) ¡Ê N
    for (int i = 0; i < 2 * height + 1; i++) {
        for (int j = 0; j < 2 * width + 1; j++) {
            kernel.at<double>(i, j) = 1.0 / ((2 * width + 1) * (2 * height + 1));
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