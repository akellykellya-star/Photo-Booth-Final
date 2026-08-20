#ifndef PHOTO_BOOTH_IMAGE_PROCESSING_HPP
#define PHOTO_BOOTH_IMAGE_PROCESSING_HPP

#include <opencv2/core.hpp>
namespace photo_booth {
cv::Mat swapRedBlueChannels(const cv::Mat& image);
/**
 * @brief Inverts every channel of an 8-bit BGR image. Each output channel value is 255 minus the corresponding input value.
 */
cv::Mat invertImage(const cv::Mat& image);
/**
 * @brief Quantizes an 8-bit BGR image to a specified number of levels.
 * Each channel is quantized independently using uniform quantization.
 */
cv::Mat quantizeImage(
    const cv::Mat& image,
    int levels);
/**
 * @brief Dynamically enhances image contrast.
 * The low and high percentiles of each channel are used as the limits of a linear contrast stretch.
 */
cv::Mat dynamicContrast(
    const cv::Mat& image,
    int low_percentile,
    int high_percentile);

/**
 * @brief Rotates an 8-bit BGR image around its center.
 * Inverse mapping and nearest-neighbor interpolation are used.
 */
cv::Mat rotateImage(
    const cv::Mat& image,
    double angle);

}

#endif
