#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat RGB_Median_Filter(Mat image, int width, int height) {
	Mat temp, output = image.clone();
	// convert to HSV so that we only need to change V
	cvtColor(image, temp, COLOR_BGR2HSV);
	cvtColor(output, output, COLOR_BGR2HSV);
	for (int i = height; i < temp.rows - height; i++) {
		for (int j = width; j < temp.cols - width; j++) {
			// compute median value to replace pixel(i, j)
			vector<uchar> window((2 * width + 1) * (2 * height + 1));
			// store all the values in the neighborhood
			for (int starty = i - height; starty <= i + height; starty++) {
				for (int startx = j - width; startx <= j + width; startx++) {
					window[(startx - j + width) + (2 * width + 1) * (starty - i + height)] = temp.at<Vec3b>(starty, startx)[2];
				}
			}
			// sort to find the median value
			sort(window.begin(), window.end());
			uchar median = window[((2 * width + 1) * (2 * height + 1) - 1) / 2];
			output.at<Vec3b>(i, j)[2] = median;
		}
	}
	cvtColor(output, output, COLOR_HSV2BGR);
	return output;
}

Mat GrayScale_Median_Filter(Mat image, int width, int height) {
	Mat output = image.clone();
	for (int i = height; i < image.rows - height; i++) {
		for (int j = width; j < image.cols - width; j++) {
			// compute median value to replace pixel(i, j)
			vector<uchar> window((2 * width + 1) * (2 * height + 1));
			// store all the values in the neighborhood
			for (int starty = i - height; starty <= i + height; starty++) {
				for (int startx = j - width; startx <= j + width; startx++) {
					window[(startx - j + width) + (2 * width + 1) * (starty - i + height)] = image.at<Vec3b>(starty, startx)[0];
				}
			}
			// sort to find the median value
			sort(window.begin(), window.end());
			uchar median = window[((2 * width + 1) * (2 * height + 1) - 1) / 2];
			output.at<Vec3b>(i, j) = Vec3b{ median, median, median };
		}
	}
	return output;
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		cerr << "MedianFilter <input-image> <output-image> <w> <h>" << endl;
		return -1;
	}
	Mat image = imread(argv[1]);	// input-image
	// Check if the image is valid
	if (image.empty()) {
		cerr << "Could not open or find the image" << endl;
		return -1;
	}
	Mat output;						// output-image
	int width = atoi(argv[3]);      // distance from anchor point to left/right border
	int height = atoi(argv[4]);     // distance from anchor point to top/bottom border	

	//medianBlur(image, output, 9);
	//imshow("example", output);

	// judge whether the input is grayscale or RGB
	if (image.channels() == 3 && image.at<Vec3b>(height, width)[0] == image.at<Vec3b>(height, width)[1] && image.at<Vec3b>(height, width)[2] == image.at<Vec3b>(height, width)[1] && image.at<Vec3b>(height, width)[0] == image.at<Vec3b>(height, width)[2]) {
		// grayscale, imread will automatically change the image to 3 channels, it's a bad way to judge yet there's no other method
		output = GrayScale_Median_Filter(image, width, height);
	} else if (image.channels() == 3) {
		// RGB
		output = RGB_Median_Filter(image, width, height);
	} else {
		// other type
		cerr << "The image is neither grayscale nor RGB." << endl;
		return -1;
	}
	
	imwrite(argv[2], output);
	imshow("input", image);
	imshow("output", output);

	waitKey(0);
	destroyWindow("input");
	destroyWindow("output");
	return 0;
}