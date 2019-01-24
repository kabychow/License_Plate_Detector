#include "include/lpd.h"
#include <iostream>

std::vector<Mat> get_dataset();
std::vector<Mat> get_test_dataset();

int main() {
    for (auto img : get_dataset()) {
        try {
            lpd result(img);
            imshow("image", result.output_image);
            waitKey();
        } catch (std::runtime_error &e) {
            std::cout << e.what();
        }
    }
    return 0;
}

std::vector<Mat> get_dataset() {
    std::vector<Mat> dataset;
    for (int i = 1; i <= 30; i++) {
        dataset.push_back(imread("dataset/" + std::to_string(i) + ".jpg"));
    }
    return dataset;
}
