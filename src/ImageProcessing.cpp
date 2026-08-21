#include "photo_booth/ImageProcessing.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace photo_booth {
namespace {
void validateImage(
    const cv::Mat& image,
    const char* function_name)
{
    if (image.empty()) {
        throw std::invalid_argument(
            std::string(function_name) +
            ": input image is empty");
    }
    if (image.type() != CV_8UC3) {
        throw std::invalid_argument(
            std::string(function_name) +
            ": input image must be CV_8UC3");
    }
}
}


cv::Mat swapRedBlueChannels(
    const cv::Mat& image)
{
    validateImage(
        image,
        "swapRedBlueChannels()");
    cv::Mat output;
    cv::cvtColor(
        image,
        output,
        cv::COLOR_BGR2RGB);
    return output;
}
cv::Mat invertImage(
    const cv::Mat& image)
{
    validateImage(
        image,
        "invertImage()");
    cv::Mat output;
    cv::bitwise_not(
        image,
        output);
    return output;
}

cv::Mat quantizeImage(
    const cv::Mat& image,
    int levels)
{
    validateImage(
        image,
        "quantizeImage()");
    if (levels <= 0 || levels > 256) {
        throw std::invalid_argument(
            "quantizeImage(): "
            "levels must be between 1 and 256");
    }

    cv::Mat output(
        image.size(),
        image.type());
    const double step =
        256.0 / static_cast<double>(levels);

    /*
     * Build a lookup table once.
     */
    unsigned char lookup[256];

    for (int value = 0;
         value < 256;
         value++) {
        const int level =
            static_cast<int>(
                value / step);
        int quantized =
            static_cast<int>(
                level * step +
                step / 2.0);
        if (quantized < 0) {
            quantized = 0;
        }
        if (quantized > 255) {
            quantized = 255;
        }
        lookup[value] =
            static_cast<unsigned char>(
                quantized);
    }

    /*
     * Apply the lookup table to every BGR channel.
     */
  
    for (int row = 0;
         row < image.rows;
         row++) {
        for (int column = 0;
             column < image.cols;
             column++) {
            const cv::Vec3b& input_pixel =
                image.at<cv::Vec3b>(
                    row,
                    column);
            cv::Vec3b& output_pixel =
                output.at<cv::Vec3b>(
                    row,
                    column);
            output_pixel[0] =
                lookup[input_pixel[0]];
            output_pixel[1] =
                lookup[input_pixel[1]];
            output_pixel[2] =
                lookup[input_pixel[2]];
        }
    }
    return output;
}

cv::Mat dynamicContrast(
    const cv::Mat& image,
    int low_percentile,
    int high_percentile)
{
    validateImage(
        image,
        "dynamicContrast()");
    if (low_percentile < 0 ||
        low_percentile >= 100) {
        throw std::invalid_argument(
            "dynamicContrast(): "
            "low percentile must be between 0 and 99");
    }
    if (high_percentile <= 0 ||
        high_percentile > 100) {
        throw std::invalid_argument(
            "dynamicContrast(): "
            "high percentile must be between 1 and 100");
    }
    if (low_percentile >= high_percentile) {

        throw std::invalid_argument(
            "dynamicContrast(): "
            "low percentile must be less than "
            "high percentile");
    }
    cv::Mat output(
        image.size(),
        image.type());
    const int total_pixels =
        image.rows * image.cols;

    /*
     * Three histograms:
     * histogram[0] = blue
     * histogram[1] = green
     * histogram[2] = red
     */
    int histogram[3][256] = {};
    for (int row = 0;
         row < image.rows;
         row++) {
        for (int column = 0;
             column < image.cols;
             column++) {
            const cv::Vec3b& pixel =
                image.at<cv::Vec3b>(
                    row,
                    column);
            histogram[0][pixel[0]]++;
            histogram[1][pixel[1]]++;
            histogram[2][pixel[2]]++;
        }
    }

    /*
     * A separate lookup table is created for each channel.
     */
    unsigned char lookup[3][256];
    for (int channel = 0;
         channel < 3;
         channel++) {
        const int low_count =
            total_pixels *
            low_percentile / 100;
        const int high_count =
            total_pixels *
            high_percentile / 100;

        int count = 0;
        int low_value = 0;
        for (int value = 0;
             value < 256;
             value++) {
            count +=
                histogram[channel][value];
            if (count > low_count) {
                low_value = value;
                break;
            }
        }

        count = 0;
        int high_value = 255;
        for (int value = 0;
             value < 256;
             value++) {
            count +=
                histogram[channel][value];
            if (count >= high_count) {
                high_value = value;
                break;
            }
        }

        if (high_value <= low_value) {
            for (int value = 0;
             value < 256;
             value++) {
            lookup[channel][value] =
                static_cast<unsigned char>(
                value);
                }
            continue;
        }

        for (int value = 0;
             value < 256;
             value++) {

            int output_value;
            if (value <= low_value) {
                output_value = 0;
            }
            else if (value >= high_value) {
                output_value = 255;
            }
            else {
                output_value =
                    static_cast<int>(
                        (value - low_value) *
                        255.0 /
                        (high_value - low_value));
            }

            if (output_value < 0) {
                output_value = 0;
            }
            if (output_value > 255) {
                output_value = 255;
            }
            lookup[channel][value] =
                static_cast<unsigned char>(
                    output_value);
        }
    }

    for (int row = 0;
         row < image.rows;
         row++) {
        for (int column = 0;
             column < image.cols;
             column++) {
            const cv::Vec3b& input_pixel =
                image.at<cv::Vec3b>(
                    row,
                    column);
            cv::Vec3b& output_pixel =
                output.at<cv::Vec3b>(
                    row,
                    column);
            output_pixel[0] =
                lookup[0][input_pixel[0]];
            output_pixel[1] =
                lookup[1][input_pixel[1]];
            output_pixel[2] =
                lookup[2][input_pixel[2]];
        }
    }

    return output;
}
cv::Mat rotateImage(
    const cv::Mat& image,
    double angle)
{
    validateImage(
        image,
        "rotateImage()");
    cv::Mat output(
        image.size(),
        image.type(),
        cv::Scalar(0, 0, 0));
    const double radians =
        angle * CV_PI / 180.0;
    const double cosine =
        std::cos(radians);
    const double sine =
        std::sin(radians);
    const double center_x =
        (image.cols - 1) / 2.0;
    const double center_y =
        (image.rows - 1) / 2.0;

    for (int row = 0;
         row < output.rows;
         row++) {
        for (int column = 0;
             column < output.cols;
             column++) {
            const double x =
                column - center_x;
            const double y =
                row - center_y;
            const double source_x =
                cosine * x +
                sine * y +
                center_x;
            const double source_y =
                -sine * x +
                cosine * y +
                center_y;
               
            const int source_column =
                static_cast<int>(
                    std::round(source_x));

            const int source_row =
                static_cast<int>(
                    std::round(source_y));

            if (source_column < 0 ||
                source_column >= image.cols ||
                source_row < 0 ||
                source_row >= image.rows) {
                continue;
            }
            output.at<cv::Vec3b>(
                row,
                column) =
                image.at<cv::Vec3b>(
                    source_row,
                    source_column);
        }
    }
    return output;
}
}
