/**
******************************************************************************
* @author  Seo HyeonGyu
* @Mod	   2525-04-18 by Seo HyeonGyu
* @brief   DLIP:  LAB - Color Image Segmentation. Make Magic Box and X-Ray
*           Magic Box is Sample 1
*           X-Ray is Sample2
*           Run the program according to the example number you want to see, and then enter the number.
*
******************************************************************************
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int hmin = 12, hmax = 46, smin = 84, smax = 167, vmin = 127, vmax = 211;
void sample1();
void sample2();

int main() {
    int num;
    cout << "Enter Sample Number (1 or 2): ";
    cin >> num;    // Run the program according to the example number you want to see, and then enter the number.

    switch (num) {
        case 1:
            sample1();
        break;
        case 2:
            sample2();
        break;
        default:
            cout << "Invalid input" << endl;
    }
    return 0;
}

void sample1() {
    Mat hsv, dst, frame, first_frame, inverted;
    vector<vector<Point>> contours;

    VideoCapture cap("../LAB_MagicCloak_Sample1.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return;
    }

    first_frame;
    cap >> first_frame; // 첫 번째 프레임을 배경으로 저장
    if (first_frame.empty()) return;


    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        imshow("Source", frame);

        cvtColor(frame, hsv, COLOR_BGR2HSV);         // Convert BGR to HSV
        inRange(hsv,         // HSV Thresholding
                Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
                Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)),
                dst);

        imshow("dsts",dst);

        // Morphological operations 노이즈 제거: Erosion -> Dilation(Opening)
        Mat erodeElem = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(dst, dst, MORPH_ERODE, erodeElem, Point(-1, -1), 3);
        Mat dilateElem = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(dst, dst, MORPH_DILATE, dilateElem, Point(-1, -1), 5);

        // Find contours
        findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat dst_out = Mat::zeros(frame.size(), CV_8UC3);

        if (!contours.empty()) {
            for (int i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i]) > 10000) // 일정 면적 이상의 외곽선만 처리 (작은 노이즈 제거)
                    drawContours(dst_out, contours, i, Scalar(255, 255, 255), FILLED, 8);
            }

            bitwise_not(dst_out, inverted);
            bitwise_and(first_frame, dst_out, dst_out);
            bitwise_and(frame, inverted, frame);
            bitwise_or(frame, dst_out, frame);
        }

        imshow("Magic_box", frame);

        char c = (char)waitKey(10);
        if (c == 27) break;
    }
}

void sample2() {
    Mat frame, hsv, mask_person, muscle_resized, bone_resized, frame_result, green_card, red_card;
    vector<vector<Point>> contours;
    Rect body_rect(0, 0, 0, 0);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return;
    }

    Mat first_frame;
    cap >> first_frame; // 배경 프레임 저장 (투명화에 활용)
    if (first_frame.empty()) return;

    Mat img_muscle = imread("../muscel.png", IMREAD_COLOR); // 근육 이미지 불러오기 및 배경 제거 (흰 배경 제거)
    Mat img_bone = imread("../bone.png", IMREAD_COLOR); // 뼈 이미지 불러오기 및 배경 제거 (흰 배경 제거)

    Mat muscle_mask, bone_mask, muscle, bone;;
    Scalar lower_white(200, 200, 200); Scalar upper_white(255, 255, 255);
    inRange(img_muscle, lower_white, upper_white, muscle_mask); // 흰색 마스크 생성
    inRange(img_bone, lower_white, upper_white, bone_mask);

    bitwise_not(muscle_mask, muscle_mask); // 역마스크: 흰 배경 제외
    bitwise_not(bone_mask, bone_mask);
    bitwise_and(img_muscle, img_muscle, muscle, muscle_mask); // 이미지에서 배경 제거
    bitwise_and(img_bone, img_bone, bone, bone_mask);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        Mat background1 = frame.clone();
        Mat background2 = frame.clone();

        cvtColor(frame, hsv, COLOR_BGR2HSV); // 사람 마스크
        inRange(hsv, Scalar(111, 82, 94), Scalar(116,106,230), mask_person);

        // 노이즈 제거 및 윤곽 강화를 위한 Dilation 연산
        Mat dilateElem = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(mask_person, mask_person, MORPH_DILATE, dilateElem, Point(-1, -1), 15);

        findContours(mask_person, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat dst_out = Mat::zeros(frame.size(), CV_8UC3);

        if (!contours.empty()) {
            for (int i = 0; i < contours.size(); i++)
                if (contourArea(contours[i]) > 65000)
                    body_rect = boundingRect(contours[i]); // 사람 윤곽 bounding box
        }

        if (body_rect.width > 0 && body_rect.height > 0) {
            double scale = 0.9; // 사람 크기에 맞춰 인체 해부도  이미지 리사이즈
            Size new_size(body_rect.width * scale, body_rect.height * scale);
            resize(muscle, muscle_resized, new_size);
            resize(bone, bone_resized, new_size);

            // 인체 해부도 이미지 위치 조정 (중앙 정렬)
            int muscle_x = body_rect.x + body_rect.width / 2 - muscle_resized.cols / 2;
            int muscle_y = body_rect.y + body_rect.height / 2 - muscle_resized.rows / 2;
            int bone_x = body_rect.x + body_rect.width / 2 - bone_resized.cols / 2;
            int bone_y = body_rect.y + body_rect.height / 2 - bone_resized.rows / 2;
            // 이미지 범위 확인 후, muscle 이미지 복사
            if (muscle_x >= 0 && muscle_y >= 0 &&
                muscle_x + muscle_resized.cols <= background1.cols &&
                muscle_y + muscle_resized.rows <= background1.rows) {

                // background1 에 근육 이미지 합성
                Rect roi(muscle_x, muscle_y, muscle_resized.cols, muscle_resized.rows);
                muscle_resized.copyTo(background1(roi));

                // 합성 시 검은 영역은 배경으로 (first_frame) 대체
                for (int y = roi.y; y < roi.y + roi.height; y++) {
                    for (int x = roi.x; x < roi.x + roi.width; x++) {
                        if (background1.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) {
                            background1.at<Vec3b>(y, x) = first_frame.at<Vec3b>(y, x);
                        }
                    }
                }
            }
            // 이미지 범위 확인 후, muscle 이미지 복사
            if (bone_x >= 0 && bone_y >= 0 &&
                bone_x + bone_resized.cols <= background2.cols &&
                bone_y + bone_resized.rows <= background2.rows){

                // background2 에 뼈 이미지 합성
                Rect roi(bone_x, muscle_y, bone_resized.cols, bone_resized.rows);
                bone_resized.copyTo(background2(roi));

                // 합성 시 검은 영역은 배경으로 (first_frame) 대체
                for (int y = roi.y; y < roi.y + roi.height; y++) {
                    for (int x = roi.x; x < roi.x + roi.width; x++) {
                        if (background2.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) {
                            background2.at<Vec3b>(y, x) = first_frame.at<Vec3b>(y, x);
                        }
                    }
                }
            }
        }
        // 초록색 카드 검출 (HSV 범위 지정)
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(50, 31, 0), Scalar(92, 219, 255), green_card);
        morphologyEx(green_card, green_card, MORPH_ERODE, dilateElem, Point(-1, -1), 5);
        morphologyEx(green_card, green_card, MORPH_DILATE, dilateElem, Point(-1, -1), 8);
        // 빨강색 카드 검출 (HSV 범위 지정)
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(150, 78, 0), Scalar(179, 219, 255), red_card);
        morphologyEx(red_card, red_card, MORPH_ERODE, dilateElem, Point(-1, -1), 5);
        morphologyEx(red_card, red_card, MORPH_DILATE, dilateElem, Point(-1, -1), 8);

        Mat frame_masked = frame.clone();
        background1.copyTo(frame_masked, green_card);// 초록색 카드 위치에만 근육 표시
        background2.copyTo(frame_masked, red_card);  // 빨강색 카드 위치에만 뼈 표시

        imshow("Card Controlled Overlay", frame_masked);
        char c = (char)waitKey(10);
        if (c == 27) break;
    }
}
