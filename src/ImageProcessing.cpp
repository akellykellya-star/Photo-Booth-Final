#include "photo_booth/ImageProcessing.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

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

}  // namespace

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
            "quantizeImage(): levels must be between 1 and 256");
    }

    cv::Mat output(
        image.size(),
        image.type());

    const double step =
        256.0 / static_cast<double>(levels);

    unsigned char lookup[256];

    for (int value = 0; value < 256; ++value) {
        const int level =
            static_cast<int>(value / step);

        int quantized =
            static_cast<int>(
                level * step + step / 2.0);

        if (quantized < 0) {
            quantized = 0;
        }

        if (quantized > 255) {
            quantized = 255;
        }

        lookup[value] =
            static_cast<unsigned char>(quantized);
    }

    for (int row = 0; row < image.rows; ++row) {
        for (int column = 0; column < image.cols; ++column) {
            const cv::Vec3b& input_pixel =
                image.at<cv::Vec3b>(row, column);

            cv::Vec3b& output_pixel =
                output.at<cv::Vec3b>(row, column);

            output_pixel[0] = lookup[input_pixel[0]];
            output_pixel[1] = lookup[input_pixel[1]];
            output_pixel[2] = lookup[input_pixel[2]];
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
            "dynamicContrast(): low percentile must be between 0 and 99");
    }

    if (high_percentile <= 0 ||
        high_percentile > 100) {
        throw std::invalid_argument(
            "dynamicContrast(): high percentile must be between 1 and 100");
    }

    if (low_percentile >= high_percentile) {
        throw std::invalid_argument(
            "dynamicContrast(): low percentile must be less than high percentile");
    }

    cv::Mat output(
        image.size(),
        image.type());

    const int total_pixels =
        image.rows * image.cols;

    int histogram[3][256] = {};

    for (int row = 0; row < image.rows; ++row) {
        for (int column = 0; column < image.cols; ++column) {
            const cv::Vec3b& pixel =
                image.at<cv::Vec3b>(row, column);

            histogram[0][pixel[0]]++;
            histogram[1][pixel[1]]++;
            histogram[2][pixel[2]]++;
        }
    }

    unsigned char lookup[3][256];

    for (int channel = 0; channel < 3; ++channel) {
        const int low_count =
            total_pixels * low_percentile / 100;

        const int high_count =
            total_pixels * high_percentile / 100;

        int count = 0;
        int low_value = 0;

        for (int value = 0; value < 256; ++value) {
            count += histogram[channel][value];

            if (count > low_count) {
                low_value = value;
                break;
            }
        }

        count = 0;
        int high_value = 255;

        for (int value = 0; value < 256; ++value) {
            count += histogram[channel][value];

            if (count >= high_count) {
                high_value = value;
                break;
            }
        }

        if (high_value <= low_value) {
            for (int value = 0; value < 256; ++value) {
                lookup[channel][value] =
                    static_cast<unsigned char>(value);
            }

            continue;
        }

        for (int value = 0; value < 256; ++value) {
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
                static_cast<unsigned char>(output_value);
        }
    }

    for (int row = 0; row < image.rows; ++row) {
        for (int column = 0; column < image.cols; ++column) {
            const cv::Vec3b& input_pixel =
                image.at<cv::Vec3b>(row, column);

            cv::Vec3b& output_pixel =
                output.at<cv::Vec3b>(row, column);

            output_pixel[0] = lookup[0][input_pixel[0]];
            output_pixel[1] = lookup[1][input_pixel[1]];
            output_pixel[2] = lookup[2][input_pixel[2]];
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

    for (int row = 0; row < output.rows; ++row) {
        for (int column = 0; column < output.cols; ++column) {
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

            output.at<cv::Vec3b>(row, column) =
                image.at<cv::Vec3b>(
                    source_row,
                    source_column);
        }
    }

    return output;
}

// Converts to grayscale while preserving the project's CV_8UC3
// BGR image representation.
cv::Mat grayscaleImage(
    const cv::Mat& image)
{
    validateImage(
        image,
        "grayscaleImage()");

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    cv::Mat output;

    cv::cvtColor(
        gray,
        output,
        cv::COLOR_GRAY2BGR);

    return output;
}

// Applies a Gaussian spatial filter.
cv::Mat blurImage(
    const cv::Mat& image,
    int kernel_size)
{
    validateImage(
        image,
        "blurImage()");

    if (kernel_size < 3 ||
        kernel_size > 31 ||
        kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "blurImage(): kernel size must be an odd value from 3 to 31");
    }

    cv::Mat output;

    cv::GaussianBlur(
        image,
        output,
        cv::Size(kernel_size, kernel_size),
        0.0);

    return output;
}

// Detects edges with Canny and converts the result back to BGR.
cv::Mat edgeDetectImage(
    const cv::Mat& image,
    double threshold1,
    double threshold2)
{
    validateImage(
        image,
        "edgeDetectImage()");

    if (threshold1 < 0.0 ||
        threshold2 <= threshold1) {
        throw std::invalid_argument(
            "edgeDetectImage(): thresholds must satisfy "
            "0 <= threshold1 < threshold2");
    }

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    cv::Mat edges;

    cv::Canny(
        gray,
        edges,
        threshold1,
        threshold2);

    cv::Mat output;

    cv::cvtColor(
        edges,
        output,
        cv::COLOR_GRAY2BGR);

    return output;
}

// Harris corner detection.
cv::Mat detectHarrisCorners(
    const cv::Mat& image,
    int blockSize,
    int apertureSize,
    double k,
    double thresholdRatio)
{
    validateImage(
        image,
        "detectHarrisCorners()");

    if (blockSize <= 0 ||
        apertureSize < 3 ||
        apertureSize > 7 ||
        apertureSize % 2 == 0 ||
        thresholdRatio < 0.0 ||
        thresholdRatio > 1.0) {
        throw std::invalid_argument(
            "detectHarrisCorners(): invalid detector parameters");
    }

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    cv::Mat grayFloat;

    gray.convertTo(
        grayFloat,
        CV_32F);

    cv::Mat response;

    cv::cornerHarris(
        grayFloat,
        response,
        blockSize,
        apertureSize,
        k);

    cv::Mat normalized;

    cv::normalize(
        response,
        normalized,
        0.0,
        255.0,
        cv::NORM_MINMAX);

    cv::Mat cornerMask;

    cv::threshold(
        normalized,
        cornerMask,
        255.0 * thresholdRatio,
        255.0,
        cv::THRESH_BINARY);

    cornerMask.convertTo(
        cornerMask,
        CV_8UC1);

    return cornerMask;
}

// FAST corner detection.
cv::Mat detectFASTCorners(
    const cv::Mat& image,
    int threshold,
    bool nonmaxSuppression)
{
    validateImage(
        image,
        "detectFASTCorners()");

    if (threshold <= 0) {
        throw std::invalid_argument(
            "detectFASTCorners(): threshold must be positive");
    }

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    const cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(
            threshold,
            nonmaxSuppression);

    std::vector<cv::KeyPoint> keypoints;

    detector->detect(
        gray,
        keypoints);

    cv::Mat cornerMask =
        cv::Mat::zeros(
            image.size(),
            CV_8UC1);

    for (const cv::KeyPoint& point : keypoints) {
        const cv::Point location(
            cvRound(point.pt.x),
            cvRound(point.pt.y));

        cv::circle(
            cornerMask,
            location,
            2,
            cv::Scalar(255),
            cv::FILLED);
    }

    return cornerMask;
}

// Draws detected corners as red circles.
cv::Mat drawCornerOverlay(
    const cv::Mat& image,
    const cv::Mat& cornerMask)
{
    validateImage(
        image,
        "drawCornerOverlay()");

    if (cornerMask.empty() ||
        cornerMask.type() != CV_8UC1 ||
        cornerMask.size() != image.size()) {
        throw std::invalid_argument(
            "drawCornerOverlay(): corner mask must be CV_8UC1 "
            "and match the input image size");
    }

    cv::Mat output =
        image.clone();

    for (int row = 0; row < cornerMask.rows; ++row) {
        for (int column = 0; column < cornerMask.cols; ++column) {
            if (cornerMask.at<unsigned char>(row, column) != 0) {
                cv::circle(
                    output,
                    cv::Point(column, row),
                    3,
                    cv::Scalar(0, 0, 255),
                    cv::FILLED);
            }
        }
    }

    return output;
}

// Creates a 48-element feature vector from three normalized
// 16-bin color histograms.
std::vector<float> extractFeatureVector(
    const cv::Mat& image)
{
    validateImage(
        image,
        "extractFeatureVector()");

    constexpr int histogramBins = 16;
    constexpr int histogramSize[] = {histogramBins};
    constexpr float histogramRange[] = {0.0f, 256.0f};

    const float* ranges[] = {histogramRange};

    std::vector<float> features;

    features.reserve(
        3 * histogramBins);

    for (int channel = 0; channel < 3; ++channel) {
        cv::Mat histogram;

        const int channels[] = {channel};

        cv::calcHist(
            &image,
            1,
            channels,
            cv::Mat(),
            histogram,
            1,
            histogramSize,
            ranges,
            true,
            false);

        cv::normalize(
            histogram,
            histogram,
            1.0,
            0.0,
            cv::NORM_L1);

        for (int bin = 0; bin < histogramBins; ++bin) {
            features.push_back(
                histogram.at<float>(bin));
        }
    }

    return features;
}

// Calculates Euclidean distance between two feature vectors.
double minimumDistance(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size()) {
        throw std::invalid_argument(
            "minimumDistance(): feature vectors must have equal "
            "nonzero sizes");
    }

    double squaredDistance = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference =
            static_cast<double>(a[i]) -
            static_cast<double>(b[i]);

        squaredDistance +=
            difference * difference;
    }

    return std::sqrt(squaredDistance);
}

// Calculates cosine similarity between two feature vectors.
double cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size()) {
        throw std::invalid_argument(
            "cosineSimilarity(): feature vectors must have equal "
            "nonzero sizes");
    }

    double dotProduct = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const double valueA = a[i];
        const double valueB = b[i];

        dotProduct += valueA * valueB;
        magnitudeA += valueA * valueA;
        magnitudeB += valueB * valueB;
    }

    if (magnitudeA == 0.0 || magnitudeB == 0.0) {
        return 0.0;
    }

    return dotProduct /
        (std::sqrt(magnitudeA) * std::sqrt(magnitudeB));
}

}  // namespace photo_booth
