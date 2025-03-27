#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

void get_gear_teeth(Mat &src, Mat &erode, Point2f &center, double &minDist);
void get_defective_gear(Mat &erode, Mat &src, Point2f &center, double &minDist);

int main() {
	Mat src, src_color, erode;
	Point2f center;
	double minDist = 1e6;

	src = imread("../../Image/Lab_GrayScale_Gears/Gear1.jpg", IMREAD_GRAYSCALE);
	cvtColor(src, src_color, COLOR_GRAY2BGR);
	//*=============================================== Gear Teeth ===============================================*//
	get_gear_teeth(src, erode, center, minDist);
	//*=========================================== Find Defective Teeth ===========================================*//
	get_defective_gear(erode, src_color, center, minDist);

	waitKey(0);
	return 0;
}

void get_gear_teeth(Mat &src, Mat &erode, Point2f &center, double &minDist) {
	vector<vector<Point>> contours;
	float radius =0.;

	// Pre-processing before find contour (Median filter % Thresholding)
	Mat src_color, blur, thresh;
	medianBlur(src, blur, 3);
	threshold(src, thresh, -1, 255, THRESH_BINARY | THRESH_OTSU );

	findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	// Get approx point entire gear
	double arclen = arcLength(contours[0], true);
	std::vector<Point2f> approx;
	approxPolyDP(contours[0], approx, 0.005 * arclen, true);

	// Get center and root diameter(=minDist) of entire gear
	minEnclosingCircle(contours[0], center, radius);
	for (size_t i = 0; i < approx.size(); i++) {
		double dist = norm(approx[i] - center);
		if (dist < minDist) minDist = dist;
	}

	// Get Image with only the gear teeth remaining by subtracting the mask image from the original image
	cvtColor(src, src_color, COLOR_GRAY2BGR);
	Mat mask = Mat::zeros(src_color.size(), CV_8UC1);
	circle(mask, center, static_cast<int>(minDist), Scalar(255), -1);

	src_color.setTo(Scalar(0, 0, 0), mask);

	// Morphology (erode)
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
	morphologyEx(src_color, erode, MORPH_ERODE, kernel, Point(-1,-1),1);
}

void get_defective_gear(Mat &erode, Mat &src, Point2f &center, double &minDist) {
	Mat blur2, thresh2;
	Mat output2, output3, output4;
	vector<vector<Point>> contours;

	cvtColor(erode, erode, COLOR_BGR2GRAY);
	output2 = Mat::zeros(src.size(), src.type());
	output3 = Mat::zeros(src.size(), src.type());
	output4 = src.clone();

	// Pre-processing before find contour (Median filter % Thresholding)
	medianBlur(erode, blur2, 3);
	threshold(blur2, thresh2, 0, 255, THRESH_BINARY | THRESH_OTSU);


	findContours(thresh2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	// Calculate average contour area and average contour length
	double totalArea = 0.0;
	double totalLength = 0.0;
	for( int i = 0; i< contours.size(); i++ ) {
		totalArea += contourArea(contours[i]);
		totalLength += arcLength(contours[i], true);
	}

	double avgArea = totalArea / contours.size();
	double avgLength = totalLength / contours.size();
	double area_percent_error   = 0.0;
	double length_percent_error = 0.0;
	int    num_defective_teeth = 0;

	for (int i = 0; i < contours.size(); i++) {
		// Calculate Area percentage error and length percentage error for each geer tooth contour
		area_percent_error = abs((contourArea(contours[i]) - avgArea) / avgArea);
		length_percent_error = abs((arcLength(contours[i], true) - avgLength) / avgLength);
		// Get contour center point
		Moments M = moments(contours[i]);
		double cx = (M.m10 / M.m00);
		double cy = (M.m01 / M.m00);
		double angle = atan2(-(cy - center.y), cx - center.x);

		// Calculate text pixel position for put Text (contour area)
		Point2f min_pos = Point2f(center.x + 0.85*minDist * cos(angle), center.y - 0.85*minDist * sin(angle));
		string text = to_string(static_cast<int>(contourArea(contours[i])));

		Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
		Point2f text_pos = min_pos - Point2f(textSize.width / 2, -textSize.height / 2 );

		// Draw contours and put Text
		if (area_percent_error < 0.09 || length_percent_error < 0.09) {
			drawContours(output2, contours, i, Scalar(0,255,0), 2);
			drawContours(output3, contours, i, Scalar(0,255,0), 2);
			putText(output2, text, text_pos ,FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
		}else {
			drawContours(output2, contours, i, Scalar(0,0,255), 2);
			drawContours(output3, contours, i, Scalar(0,0,255), 2);
			putText(output2, text, text_pos,FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
			num_defective_teeth++;

			int dash_length = 5;
			int gap_length = 20;
			for (int theta = 0; theta < 360; theta += dash_length + gap_length) {
				double rad1 = theta * CV_PI / 180.0;
				double rad2 = (theta + dash_length) * CV_PI / 180.0;

				Point2d pt1(cx + 30 * cos(rad1), cy + 30 * sin(rad1));
				Point2d pt2(cx + 30 * cos(rad2), cy + 30 * sin(rad2));

				line(output4, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
			}
		}
	}

	// Output the final result status
	string Quality;
	Quality = (num_defective_teeth > 0) ? "FAIL" : "PASS";
	char buffer[200];

	sprintf(buffer, "Teeth numbers: %d \nDiameter of Gear: %.2f\nAvg. Teeth Area: %.2f\nDefective Teeth: %d\nQuality: %s",
						(int)contours.size(), minDist, avgArea, num_defective_teeth, Quality.c_str());


	stringstream ss(buffer);
	string line;
	int y_offset = src.rows/2 + minDist/2.5; // 줄 간격 조정

	while (getline(ss, line, '\n')) {
		putText(output4, line, Point(src.cols/2 + 1.3*minDist , y_offset), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
		y_offset += 20;
	}

	cout << "Teeth numbers: " << (int)contours.size() << endl;
	cout << "Diameter of Gear: " << minDist << endl;
	cout << "Avg. Teeth Area: " << avgArea << endl;
	cout << "Defective Teeth: " << num_defective_teeth << endl;
	cout << "Quality: " << Quality.c_str() << endl;

	imshow("Original", src);
	imshow("Output2", output2);
	imshow("Output3", output3);
	imshow("Output4", output4);
}
