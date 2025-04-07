#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat src, src_gray, src_crop;
Mat Gauss, edge;
Mat left_coeff, right_coeff;

#define PI 3.1415926535897932384626433832795

/**
 * @function polyFit
 * @brief The lane model is calculated through polynomial regression using points detected through houghlinesP.
 */
Mat polyFit(const cv::Mat& X_, const cv::Mat& Y_, int n) {
    if (X_.rows != Y_.rows || X_.cols != 1 || Y_.cols != 1) {
        std::cerr << "X and Y must be column vectors with the same number of rows." << std::endl;
        return Mat();
    }

    Mat X2, Y2;
    X_.convertTo(X2, CV_64F);
    Y_.convertTo(Y2, CV_64F);

    Mat A(X2.rows, n + 1, CV_64F);
    for (int i = 0; i < X2.rows; ++i) {
        double x_val = X2.at<double>(i, 0);
        for (int j = 0; j < n + 1; ++j) {
            A.at<double>(i, j) = std::pow(x_val, n - j);
        }
    }

    Mat At = A.t();
    Mat AtA, AtY, c;

    AtA = At * A;
    AtY = At * Y2;
    solve(AtA, AtY, c, cv::DECOMP_SVD);
    return c;
}

/**
 * @function preprocess
 * @brief Image preprocessing for efficient lane detection Gaussian Blur & Canny edge detection
 */
void preprocess(const int roi_row) {
    Rect roi(0,roi_row,src.cols,src.rows-roi_row);
    Mat src_crop = src(roi); // Crop the area around the lane as the region of interest.

    cvtColor(src_crop, src_gray, COLOR_BGR2GRAY); // Convert to gray for Gaussian blur application.

    GaussianBlur(src_gray, Gauss, Size(3, 3), 1, 1); // Blurring with kernel size (3,3)/ sigma = 1
    Canny(Gauss, edge, 240,250, 3); // Canny Edge Detection LowT = 240, HighT = 250 (Hysteresis thresholding)
}
/**
 * @function lane_detection
 * @brief Lane detection via houghlinesP and one line detection per lane via polyFit.
 */
void lane_detection(Mat& left_coeff, Mat& right_coeff, const int roi_row) {
    // 1. 차선을 담을 선분 벡터와, 왼쪽/오른쪽 차선 좌표를 담을 벡터 선언
    vector<Vec4i> linesP;
    vector<Point> leftPoints, rightPoints;
    double angle = 0.;

    // 2. Hough 변환을 통해 직선 검출
    HoughLinesP(edge, linesP, 1, CV_PI / 180, 14, 3, 3);
    //  - 14: 투표 임계값
    //  - 3, 3: 최소 선 길이, 최대 허용 간격

    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];

        angle = atan2(l[3] - l[1], l[2] - l[0]) * 180 / PI;

        // 수평에 가까운 선은 무시
        if (abs(angle) < 20) continue;

        // 기울기 방향에 따라 왼쪽 또는 오른쪽 차선으로 분류
        if (angle < 0) {
            leftPoints.emplace_back(l[0], l[1] + roi_row);
            leftPoints.emplace_back(l[2], l[3] + roi_row);
        } else {
            rightPoints.emplace_back(l[0], l[1] + roi_row);
            rightPoints.emplace_back(l[2], l[3] + roi_row);
        }
    }

    // 3. Linear regression 수행
    Mat leftX, leftY;
    for (const Point& pt : leftPoints) {
        leftX.push_back(pt.x);
        leftY.push_back(pt.y);
    }
    left_coeff = polyFit(leftX, leftY, 1);

    Mat rightX, rightY;
    for (const Point& pt : rightPoints) {
        rightX.push_back(pt.x);
        rightY.push_back(pt.y);
    }
    right_coeff = polyFit(rightX, rightY, 1);
}

/**
 * @function draw_lane
 * @brief Calculate Vanishing Point and Draw lane and vanishing point on source image.
 */
void draw_lane() {
    // Calculate the vanishing point (intersection of left and right lane lines)
    double vanishing_x = (right_coeff.at<double>(1) - left_coeff.at<double>(1)) /
                         (left_coeff.at<double>(0) - right_coeff.at<double>(0));
    double vanishing_y = left_coeff.at<double>(0) * vanishing_x + left_coeff.at<double>(1);

    Point vanishing_point(cvRound(vanishing_x), cvRound(vanishing_y));

    // Calculate x-coordinates of where the lane lines touch the bottom of the image
    double left_line_x = (src.rows - left_coeff.at<double>(1)) / left_coeff.at<double>(0);
    double right_line_x = (src.rows - right_coeff.at<double>(1)) / right_coeff.at<double>(0);

    Point left_start(cvRound(left_line_x), src.rows);
    Point right_start(cvRound(right_line_x), src.rows);

    // Calculate the horizontal gaps between the vanishing point and lane lines
    double left_gap = vanishing_x - left_start.x;
    double right_gap = right_start.x - vanishing_x;

    Scalar color;
    if (left_gap > 3 * right_gap || right_gap > 3 * left_gap) { // Lane departure detected (car is too far from center)
        color = Scalar(116, 69, 230);
        putText(src, "Left the lane", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    }
    else { // Car is within the lane boundaries
        color = Scalar(79, 193, 71);
        putText(src, "Stay in the lane", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(50, 89, 40), 2);
    }

    Mat load = src.clone();
    vector road_points = { left_start, vanishing_point, right_start };

    fillPoly(load, vector<vector<Point>>{road_points}, color);
    double alpha = 0.4;
    addWeighted(load, alpha, src, 1 - alpha, 0, src); // Fill the lane area with transparent color

    line(src, left_start, vanishing_point, Scalar(0, 0, 255), 2, 1);       // Left line
    line(src, right_start, vanishing_point, Scalar(0, 255, 0), 2, 1);      // Right line
    circle(src, vanishing_point, 7, Scalar(193, 120, 255), 2);
    line(src, vanishing_point, Point(vanishing_point.x, src.rows), Scalar(255, 0, 0), 1, 1); // Vertical guide line
}


/**
 * @function main
 */
int main()
{
    String filename = "../Lane_changing.jpg";
    src = imread(filename, IMREAD_COLOR);
    int roi_rows = 3 * src.rows/5;

    //*=============================================== Pre-processing ===============================================*//
    preprocess(roi_rows);

    //*=============================================== Lane Detection ===============================================*//
    lane_detection(left_coeff, right_coeff, roi_rows);

    //*========================================= Draw Lane & Vanishing Point ========================================*//
    if (!left_coeff.empty() && !right_coeff.empty()) draw_lane();

    imshow("src", src);
    waitKey(0);
    return 0;
}

