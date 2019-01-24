#ifndef LICENSE_PLATE_RECOGNITION_FUNCTION_H
#define LICENSE_PLATE_RECOGNITION_FUNCTION_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;

Mat rgb_to_grey(Mat img) {
    Mat output = Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols*3; j+=3) {
            int avg = (img.at<uchar>(i, j) + img.at<uchar>(i, j + 1) + img.at<uchar>(i, j + 2)) / 3;
            output.at<uchar>(i, j / 3) = avg;
        }
    }
    return output;
}

Mat grey_equalize(Mat img) {
    Mat output = Mat::zeros(img.size(), CV_8UC1);
    double count[256] = {0}, calculated[256] = {0}, accumulate = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            count[img.at<uchar>(i, j)]++;
        }
    }
    for (int i = 0; i < 256; i++) {
        accumulate += (count[i] / (img.rows * img.cols));
        calculated[i] = accumulate;
        calculated[i] *= 255;
        calculated[i] = round(calculated[i]);
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            output.at<uchar>(i, j) = calculated[img.at<uchar>(i, j)];
        }
    }
    return output;
}

Mat grey_blur(Mat img, int level) {
    Mat output = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            int total = 0;
            for (int ii = i-level; ii <= i+level; ii++) {
                for (int jj = j-level; jj <= j+level; jj++) {
                    total += img.at<uchar>(ii, jj);
                }
            }
            output.at<uchar>(i-1, j-1) = total / pow((level * 2 + 1), 2);
        }
    }
    return output;
}

Mat rgb_crop(Mat img, double percentage_up, double percentage_right, double percentage_down, double percentage_left) {
    Mat output = Mat::zeros(img.rows * (1 - percentage_up - percentage_down), img.cols * (1 - percentage_left - percentage_right), CV_8UC3);
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols * 3; j++) {
            output.at<uchar>(i, j) = img.at<uchar>(i + (percentage_up * img.rows), j + (percentage_left * (img.cols * 3)));
        }
    }
    return output;
}

Mat rgb_expand_top(Mat img, int level) {
    Mat output = Mat::zeros(img.rows + level, img.cols, CV_8UC3);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols * 3; j++) {
            output.at<uchar>(i + level, j) = img.at<uchar>(i, j);
        }
    }
    return output;
}

Mat grey_vertical_edge(Mat img, int threshold) {
    Mat output = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            int x = (img.at<uchar>(i-1, j-1) + img.at<uchar>(i, j-1) + img.at<uchar>(i+1, j-1)) / 3;
            int y = (img.at<uchar>(i-1, j+1) + img.at<uchar>(i, j+1) + img.at<uchar>(i+1, j+1)) / 3;
            if (abs(x - y) >= threshold) output.at<uchar>(i-1, j-1) = 255;
        }
    }
    return output;
}

Mat grey_dilation(Mat img, int level) {
    Mat output = Mat::zeros(img.rows - 2 * level, img.cols - 2 * level, CV_8UC1);
    for (int i = level; i < img.rows - level; i++) {
        for (int j = level; j < img.cols - level; j++) {
            for (int t_i = -level; t_i <= level; t_i++) {
                for (int t_j = -level; t_j <= level; t_j++) {
                    if (img.at<uchar>(i + t_i, j + t_j) == 255) output.at<uchar>(i-level, j-level) = 255;
                }
            }
        }
    }
    return output;
}

Mat grey_binarize(Mat img, int threshold) {
    Mat output = Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) >= threshold) output.at<uchar>(i, j) = 255;
        }
    }
    return output;
}

int grey_get_otsu(Mat img) {
    Mat output = Mat::zeros(img.size(), CV_8UC1);
    double count[256] = {0}, probability[256] = {0}, accumulate_probability[256] = {0}, meu[256] = {0};
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            count[img.at<uchar>(i, j)]++;
        }
    }
    for (int i = 0; i < 256; i++) {
        probability[i] = count[i] / (img.rows * img.cols);
        accumulate_probability[i] = (i == 0) ? probability[i] : accumulate_probability[i-1] + probability[i];
        meu[i] = (i == 0) ? i * probability[i] : meu[i-1] + i * probability[i];
    }
    double max_sigma = -1, max_sigma_otsu = -1;
    for (int i = 0; i < 256; i++) {
        double sigma = pow(meu[255] * accumulate_probability[i] - meu[i], 2) / (accumulate_probability[i] * (1 - accumulate_probability[i]));
        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_sigma_otsu = i;
        }
    }
    return max_sigma_otsu;
}

double grey_get_density(Mat img) {
    int count = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) > 0) count++;
        }
    }
    return (double)count / (double)(img.rows * img.cols);
}

Mat rgb_draw_rect(Mat img, Rect rect) {
    for (int i = rect.y - 4; i < rect.y + rect.height + 4; i++) {
        for (int j = rect.x - 4; j < (rect.x + rect.width + 4); j++) {
            if (i < rect.y || i > rect.y + rect.height || j < rect.x || j > rect.x + rect.width) {
                img.at<uchar>(i, j*3) = 0;
                img.at<uchar>(i, j*3+1) = 255;
                img.at<uchar>(i, j*3+2) = 0;
            }
        }
    }
    return img;
}

std::vector<Rect> find_contours(Mat img) {
    std::vector<std::vector<Point>> raw_contours;
    std::vector<Rect> contours;
    findContours(img, raw_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (auto contour : raw_contours) {
        contours.push_back(boundingRect(contour));
    }
    return contours;
}

#endif
