#ifndef PHOTO_BOOTH_IMAGE_PROCESSING_HPP
#define PHOTO_BOOTH_IMAGE_PROCESSING_HPP

#include <opencv2/core.hpp>
namespace photo_booth {

cv::Mat swapRedBlueChannels(
    const cv::Mat& image);

cv::Mat invertImage(
    const cv::Mat& image);

cv::Mat quantizeImage(
    const cv::Mat& image,
    int levels);

cv::Mat dynamicContrast(
    const cv::Mat& image,
    int low_percentile,
    int high_percentile);

cv::Mat rotateImage(
    const cv::Mat& image,
    double angle);

}
