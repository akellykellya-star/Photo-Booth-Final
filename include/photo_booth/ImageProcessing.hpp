#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace photo_booth {

cv::Mat swapRedBlueChannels(const cv::Mat& image);
cv::Mat invertImage(const cv::Mat& image);
cv::Mat quantizeImage(const cv::Mat& image, int levels);
cv::Mat igsQuantizeImage(const cv::Mat& image, int levels = 16);
cv::Mat otsuThresholdImage(const cv::Mat& image);
cv::Mat dynamicContrast(const cv::Mat& image, int low_percentile, int high_percentile);
cv::Mat rotateImage(const cv::Mat& image, double angle);
cv::Mat grayscaleImage(const cv::Mat& image);
cv::Mat blurImage(const cv::Mat& image, int kernel_size);
cv::Mat edgeDetectImage(const cv::Mat& image, double threshold1, double threshold2);

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

cv::Mat embedIconPhaseDispersion(const cv::Mat& image);
cv::Mat extractEmbeddedIcon(const cv::Mat& image);

cv::Mat linearContrast(const cv::Mat& image, double inputLow, double inputHigh);
cv::Mat histogramEqualization(const cv::Mat& image);
cv::Mat histogramMatching(const cv::Mat& image, const cv::Mat& reference);

cv::Mat affineTransform(const cv::Mat& image, double angleDegrees,
                        double scale, double translateX, double translateY);
cv::Mat perspectiveTransform(const cv::Mat& image,
                             const std::vector<cv::Point2f>& sourcePoints,
                             const std::vector<cv::Point2f>& destinationPoints);
cv::Mat logPolarTransform(const cv::Mat& image, double magnitude = 40.0);

cv::Mat spatialFilter(const cv::Mat& image, int preset, int kernelSize);
cv::Mat customSpatialFilter(const cv::Mat& image, const cv::Mat& kernel);

cv::Mat frequencyFilter(const cv::Mat& image, bool lowPass, int shape,
                        double cutoff, int butterworthOrder = 2);

cv::Mat imageFeatureVector(const cv::Mat& image);
struct ClassificationResult {
    int label{-1};
    double score{0.0};
};
ClassificationResult minimumDistanceClassifier(
    const cv::Mat& feature, const std::vector<cv::Mat>& classSamples,
    const std::vector<int>& labels);
ClassificationResult cosineSimilarityClassifier(
    const cv::Mat& feature, const std::vector<cv::Mat>& classSamples,
    const std::vector<int>& labels);
double classificationAccuracy(
    const std::vector<cv::Mat>& samples,
    const std::vector<int>& labels,
    bool useCosineSimilarity);

}  // namespace photo_booth
