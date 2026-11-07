#pragma once

#include <opencv2/core.hpp>

#include <vector>

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

// Week 15 additions.

cv::Mat grayscaleImage(
    const cv::Mat& image);

cv::Mat blurImage(
    const cv::Mat& image,
    int kernel_size);

cv::Mat edgeDetectImage(
    const cv::Mat& image,
    double threshold1,
    double threshold2);

// Harris corner detection.
cv::Mat detectHarrisCorners(
    const cv::Mat& image,
    int blockSize = 2,
    int apertureSize = 3,
    double k = 0.04,
    double thresholdRatio = 0.01);

// FAST corner detection.
cv::Mat detectFASTCorners(
    const cv::Mat& image,
    int threshold = 20,
    bool nonmaxSuppression = true);

// Draws detected corners over the original image.
cv::Mat drawCornerOverlay(
    const cv::Mat& image,
    const cv::Mat& cornerMask);

// Extracts a normalized color histogram feature vector.
std::vector<float> extractFeatureVector(
    const cv::Mat& image);

// Minimum-distance classifier.
double minimumDistance(
    const std::vector<float>& a,
    const std::vector<float>& b);

// Cosine-similarity classifier.
double cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b);

}
