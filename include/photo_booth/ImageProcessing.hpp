#pragma once

#include <opencv2/core.hpp>

namespace photo_booth {

cv::Mat swapRedBlueChannels(const cv::Mat& image);
cv::Mat invertImage(const cv::Mat& image);
cv::Mat quantizeImage(const cv::Mat& image, int levels);
cv::Mat dynamicContrast(const cv::Mat& image, int low_percentile, int high_percentile);
cv::Mat rotateImage(const cv::Mat& image, double angle);
cv::Mat grayscaleImage(const cv::Mat& image);
cv::Mat blurImage(const cv::Mat& image, int kernel_size);
cv::Mat edgeDetectImage(const cv::Mat& image, double threshold1, double threshold2);

// Corner masks are single-channel images. The overlay keeps the original
// three-channel BGR frame and marks detected corners in red.
cv::Mat detectHarrisCorners(
    const cv::Mat& image,
    int blockSize = 2,
    int apertureSize = 3,
    double k = 0.04,
    double thresholdRatio = 0.01);

cv::Mat detectFASTCorners(
    const cv::Mat& image,
    int threshold = 20,
    bool nonmaxSuppression = true);

cv::Mat drawCornerOverlay(
    const cv::Mat& image,
    const cv::Mat& cornerMask);

// Embed a small built-in camera icon in the luminance-channel Fourier phase.
// The icon can be recovered from the resulting image with extractEmbeddedIcon.
cv::Mat embedIconPhaseDispersion(const cv::Mat& image);
cv::Mat extractEmbeddedIcon(const cv::Mat& image);

}  // namespace photo_booth
