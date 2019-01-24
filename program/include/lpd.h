#ifndef LICENSE_PLATE_RECOGNITION_LPD_H
#define LICENSE_PLATE_RECOGNITION_LPD_H

#include "function.h"
#include "neural_network.h"

neural_network nn;

struct lpd {
    Mat image;
    Mat plate;
    std::vector<Mat> chars;

    Mat output_image;
    std::string output_text;

    lpd(Mat image) {
        this->image = image;
        find_plate();
        plate_find_chars();
        plate_read_chars();
    }

    void find_plate() {
        Mat crop = rgb_crop(image, 0.3, 0.1, 0, 0.2);
        Mat grey = rgb_to_grey(crop);
        Mat equalize = grey_equalize(grey);
        Mat blur = grey_blur(equalize, 1);
        for (int edge_threshold_step = -8; edge_threshold_step <= 8; edge_threshold_step += 2 * 8) {
            for (int edge_threshold = 43; edge_threshold >= 0 && edge_threshold <= 255; edge_threshold += edge_threshold_step) {
                Mat edge = grey_vertical_edge(blur, edge_threshold);
                Mat dilation = grey_dilation(edge, 4);
                std::vector<Rect> matches;
                for (auto blob : find_contours(dilation)) {
                    if (blob.height >= 24 && blob.width <= 200 &&
                        blob.width >= blob.height * 1.35 && blob.width <= blob.height * 4 &&
                        grey_get_density(dilation(blob)) >= 0.7) {
                        matches.push_back(blob);
                    }
                }
                if (matches.size() > 0) {
                    Rect best_match = matches.front();
                    for (auto match : matches) {
                        if ((double)match.width / (double)match.height > (double)best_match.width / (double)best_match.height) {
                            best_match = match;
                        }
                    }
                    best_match.x += image.cols * 0.2 + 10;
                    best_match.y += image.rows * 0.3 + 8;
                    best_match.width += 4;
                    best_match.height -= 6;
                    output_image = rgb_draw_rect(image, best_match);
                    plate = image(best_match);
                    return;
                }
            }
        }
        throw std::runtime_error("\nCould not find plate");
    }

    void plate_find_chars() {
        plate = rgb_to_grey(plate);
        for (int binarize_threshold_step = -15; binarize_threshold_step <= 15; binarize_threshold_step += 2 * 15) {
            for (int binarize_threshold = grey_get_otsu(plate) + 73; binarize_threshold >= 0 && binarize_threshold <= 255; binarize_threshold += binarize_threshold_step) {
                Mat binarized_plate = grey_binarize(plate, binarize_threshold);
                std::vector<Rect> matches;
                for (auto blob : find_contours(binarized_plate)) {
                    if (blob.height >= 10 && blob.height <= 30 && blob.width >= 2 && blob.width <= 30) {
                        int size = static_cast<int>(matches.size());
                        for (int i = 0; i < size && matches.size() <= size; i++) {
                            if (abs(matches.at(i).y - blob.y) < 10 && blob.x <= matches.at(i).x) {
                                matches.insert(matches.begin() + i, blob);
                            } else if (matches.at(i).y - blob.y > 10) {
                                matches.insert(matches.begin(), blob);
                            }
                        }
                        if (matches.size() == size) matches.push_back(blob);
                    }
                }
                if (matches.size() > chars.size() && matches.size() <= 7) {
                    chars.clear();
                    for (auto match : matches) {
                        chars.push_back(binarized_plate(match));
                    }
                }
            }
            if (chars.size() >= 4) break;
        }
    }

    void plate_read_chars() {
        output_text = "";
        for (auto c : chars) {
            nn.set_image(c);
            output_text += nn.predict();
        }
        output_image = rgb_expand_top(image, 40);
        putText(output_image, output_text, Point(10, 30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255));
    }
};


#endif
