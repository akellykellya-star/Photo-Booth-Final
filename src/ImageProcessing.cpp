#include "photo_booth/ImageProcessing.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <limits>

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

cv::Mat swapRedBlueChannels(const cv::Mat& image)
{
    validateImage(image, "swapRedBlueChannels()");
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_BGR2RGB);
    return output;
}

cv::Mat invertImage(const cv::Mat& image)
{
    validateImage(image, "invertImage()");
    cv::Mat output;
    cv::bitwise_not(image, output);
    return output;
}

cv::Mat quantizeImage(const cv::Mat& image, int levels)
{
    validateImage(image, "quantizeImage()");
    if (levels <= 0 || levels > 256) {
        throw std::invalid_argument(
            "quantizeImage(): levels must be between 1 and 256");
    }

    cv::Mat output(image.size(), image.type());
    const double step = 256.0 / static_cast<double>(levels);
    unsigned char lookup[256];

    for (int value = 0; value < 256; ++value) {
        const int level = static_cast<int>(value / step);
        int quantized = static_cast<int>(level * step + step / 2.0);

        if (quantized < 0) quantized = 0;
        if (quantized > 255) quantized = 255;

        lookup[value] = static_cast<unsigned char>(quantized);
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
    validateImage(image, "dynamicContrast()");

    if (low_percentile < 0 || low_percentile >= 100 ||
        high_percentile <= 0 || high_percentile > 100 ||
        low_percentile >= high_percentile) {
        throw std::invalid_argument(
            "dynamicContrast(): invalid percentile range");
    }

    cv::Mat output(image.size(), image.type());
    const int total_pixels = image.rows * image.cols;
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
                output_value = static_cast<int>(
                    (value - low_value) * 255.0 /
                    (high_value - low_value));
            }

            if (output_value < 0) output_value = 0;
            if (output_value > 255) output_value = 255;

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

cv::Mat rotateImage(const cv::Mat& image, double angle)
{
    validateImage(image, "rotateImage()");
    cv::Mat output(image.size(), image.type(), cv::Scalar(0, 0, 0));

    const double radians = angle * CV_PI / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    const double center_x = (image.cols - 1) / 2.0;
    const double center_y = (image.rows - 1) / 2.0;

    for (int row = 0; row < output.rows; ++row) {
        for (int column = 0; column < output.cols; ++column) {
            const double x = column - center_x;
            const double y = row - center_y;
            const double source_x =
                cosine * x + sine * y + center_x;
            const double source_y =
                -sine * x + cosine * y + center_y;

            const int source_column =
                static_cast<int>(std::round(source_x));
            const int source_row =
                static_cast<int>(std::round(source_y));

            if (source_column < 0 || source_column >= image.cols ||
                source_row < 0 || source_row >= image.rows) {
                continue;
            }

            output.at<cv::Vec3b>(row, column) =
                image.at<cv::Vec3b>(source_row, source_column);
        }
    }

    return output;
}

cv::Mat grayscaleImage(const cv::Mat& image)
{
    validateImage(image, "grayscaleImage()");
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat output;
    cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
    return output;
}

cv::Mat blurImage(const cv::Mat& image, int kernel_size)
{
    validateImage(image, "blurImage()");

    if (kernel_size < 3 || kernel_size > 31 ||
        kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "blurImage(): kernel size must be an odd value from 3 to 31");
    }

    cv::Mat output;
    cv::GaussianBlur(
        image, output, cv::Size(kernel_size, kernel_size), 0.0);
    return output;
}

cv::Mat edgeDetectImage(
    const cv::Mat& image,
    double threshold1,
    double threshold2)
{
    validateImage(image, "edgeDetectImage()");

    if (threshold1 < 0.0 || threshold2 <= threshold1) {
        throw std::invalid_argument(
            "edgeDetectImage(): thresholds must satisfy "
            "0 <= threshold1 < threshold2");
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, threshold1, threshold2);

    cv::Mat output;
    cv::cvtColor(edges, output, cv::COLOR_GRAY2BGR);
    return output;
}

cv::Mat detectHarrisCorners(
    const cv::Mat& image,
    int blockSize,
    int apertureSize,
    double k,
    double thresholdRatio)
{
    validateImage(image, "detectHarrisCorners()");

    if (blockSize < 2 || apertureSize < 3 ||
        apertureSize > 7 || apertureSize % 2 == 0 ||
        k <= 0.0 || thresholdRatio < 0.0 ||
        thresholdRatio > 1.0) {
        throw std::invalid_argument(
            "detectHarrisCorners(): invalid Harris parameters");
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat response;
    cv::cornerHarris(gray, response, blockSize, apertureSize, k);

    double maximum_response = 0.0;
    cv::minMaxLoc(response, nullptr, &maximum_response);

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    if (maximum_response <= 0.0) {
        return mask;
    }

    const float threshold =
        static_cast<float>(maximum_response * thresholdRatio);

    for (int row = 0; row < response.rows; ++row) {
        for (int column = 0; column < response.cols; ++column) {
            if (response.at<float>(row, column) > threshold) {
                mask.at<unsigned char>(row, column) = 255;
            }
        }
    }

    return mask;
}

cv::Mat detectFASTCorners(
    const cv::Mat& image,
    int threshold,
    bool nonmaxSuppression)
{
    validateImage(image, "detectFASTCorners()");

    if (threshold <= 0) {
        throw std::invalid_argument(
            "detectFASTCorners(): threshold must be positive");
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, threshold, nonmaxSuppression);

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    for (const auto& keypoint : keypoints) {
        const cv::Point point(
            cvRound(keypoint.pt.x),
            cvRound(keypoint.pt.y));

        if (point.x >= 0 && point.x < mask.cols &&
            point.y >= 0 && point.y < mask.rows) {
            mask.at<unsigned char>(point) = 255;
        }
    }

    return mask;
}
cv::Mat drawCornerOverlay(
    const cv::Mat& image,
    const cv::Mat& cornerMask)
{
    validateImage(image, "drawCornerOverlay()");

    if (cornerMask.empty() || cornerMask.size() != image.size() ||
        cornerMask.type() != CV_8UC1) {
        throw std::invalid_argument(
            "drawCornerOverlay(): corner mask must be a matching CV_8UC1 image");
    }

    cv::Mat output = image.clone();

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int componentCount = cv::connectedComponentsWithStats(
        cornerMask,
        labels,
        stats,
        centroids);

    for (int i = 1; i < componentCount; ++i) {
        const int centerX = static_cast<int>(centroids.at<double>(i, 0));
        const int centerY = static_cast<int>(centroids.at<double>(i, 1));

        cv::circle(
            output,
            cv::Point(centerX, centerY),
            4,
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA);
    }

    return output;
}

namespace {

cv::Mat makeCameraIcon()
{
    cv::Mat icon = cv::Mat::zeros(16, 16, CV_8UC1);
    cv::rectangle(icon, cv::Rect(2, 5, 12, 9), cv::Scalar(255), 1);
    cv::rectangle(icon, cv::Rect(5, 3, 5, 2), cv::Scalar(255), -1);
    cv::circle(icon, cv::Point(8, 9), 3, cv::Scalar(255), 1);
    cv::circle(icon, cv::Point(8, 9), 1, cv::Scalar(255), -1);
    return icon;
}

cv::Mat getLuminance(const cv::Mat& image)
{
    validateImage(image, "phase-dispersion operation");

    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    return channels[0];
}

cv::Mat paddedComplexDft(const cv::Mat& gray)
{
    cv::Mat float_image;
    gray.convertTo(float_image, CV_32F);

    const int rows = cv::getOptimalDFTSize(float_image.rows);
    const int cols = cv::getOptimalDFTSize(float_image.cols);

    cv::Mat padded;
    cv::copyMakeBorder(
        float_image, padded, 0, rows - float_image.rows,
        0, cols - float_image.cols, cv::BORDER_CONSTANT,
        cv::Scalar::all(0));

    cv::Mat planes[] = {
        padded,
        cv::Mat::zeros(padded.size(), CV_32F)
    };

    cv::Mat complex_image;
    cv::merge(planes, 2, complex_image);
    cv::dft(complex_image, complex_image);
    return complex_image;
}

}  // namespace

cv::Mat embedIconPhaseDispersion(const cv::Mat& image)
{
    validateImage(image, "embedIconPhaseDispersion()");

    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    cv::Mat complex_image = paddedComplexDft(channels[0]);
    const int rows = complex_image.rows;
    const int cols = complex_image.cols;

    if (rows <= 32 || cols <= 32) {
        throw std::invalid_argument(
            "embedIconPhaseDispersion(): image is too small for the icon payload");
    }

    const cv::Mat icon = makeCameraIcon();
    const double minimum_magnitude = 150000.0;

    for (int icon_y = 0; icon_y < icon.rows; ++icon_y) {
        for (int icon_x = 0; icon_x < icon.cols; ++icon_x) {
            const int frequency_y = 2 + icon_y * 2;
            const int frequency_x = 2 + icon_x * 2;
            const int mirror_y = (rows - frequency_y) % rows;
            const int mirror_x = (cols - frequency_x) % cols;

            cv::Vec2f& coefficient =
                complex_image.at<cv::Vec2f>(frequency_y, frequency_x);

            const double magnitude = std::max(
                minimum_magnitude,
                static_cast<double>(
                    std::hypot(coefficient[0], coefficient[1])));

            const bool bit =
                icon.at<unsigned char>(icon_y, icon_x) > 0;
            const double phase = bit ? 0.65 : -0.65;

            coefficient[0] =
                static_cast<float>(magnitude * std::cos(phase));
            coefficient[1] =
                static_cast<float>(magnitude * std::sin(phase));

            if (mirror_y != frequency_y || mirror_x != frequency_x) {
                cv::Vec2f& mirror =
                    complex_image.at<cv::Vec2f>(mirror_y, mirror_x);
                mirror[0] = coefficient[0];
                mirror[1] = -coefficient[1];
            }
        }
    }

    cv::Mat reconstructed;
    cv::dft(
        complex_image,
        reconstructed,
        cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    reconstructed(
        cv::Rect(0, 0, image.cols, image.rows)).convertTo(
            channels[0], CV_8U);

    cv::merge(channels, ycrcb);

    cv::Mat output;
    cv::cvtColor(ycrcb, output, cv::COLOR_YCrCb2BGR);
    return output;
}

cv::Mat extractEmbeddedIcon(const cv::Mat& image)
{
    const cv::Mat gray = getLuminance(image);
    const cv::Mat complex_image = paddedComplexDft(gray);

    if (complex_image.rows <= 32 || complex_image.cols <= 32) {
        throw std::invalid_argument(
            "extractEmbeddedIcon(): image is too small for the icon payload");
    }

    cv::Mat icon = cv::Mat::zeros(16, 16, CV_8UC1);

    for (int icon_y = 0; icon_y < icon.rows; ++icon_y) {
        for (int icon_x = 0; icon_x < icon.cols; ++icon_x) {
            const int frequency_y = 2 + icon_y * 2;
            const int frequency_x = 2 + icon_x * 2;

            const cv::Vec2f coefficient =
                complex_image.at<cv::Vec2f>(frequency_y, frequency_x);

            const double phase =
                std::atan2(coefficient[1], coefficient[0]);

            icon.at<unsigned char>(icon_y, icon_x) =
                phase >= 0.0 ? 255 : 0;
        }
    }

    return icon;
}


cv::Mat linearContrast(const cv::Mat& image, double inputLow, double inputHigh)
{
    validateImage(image, "linearContrast()");
    if (inputLow < 0.0 || inputHigh > 255.0 || inputHigh <= inputLow)
        throw std::invalid_argument("linearContrast(): invalid input range");
    cv::Mat output;
    image.convertTo(output, CV_32FC3,
        255.0 / (inputHigh - inputLow),
        -inputLow * 255.0 / (inputHigh - inputLow));
    output.convertTo(output, CV_8UC3);
    return output;
}

cv::Mat histogramEqualization(const cv::Mat& image)
{
    validateImage(image, "histogramEqualization()");
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    cv::equalizeHist(channels[0], channels[0]);
    cv::merge(channels, ycrcb);
    cv::Mat output;
    cv::cvtColor(ycrcb, output, cv::COLOR_YCrCb2BGR);
    return output;
}

cv::Mat histogramMatching(const cv::Mat& image, const cv::Mat& reference)
{
    validateImage(image, "histogramMatching()");
    validateImage(reference, "histogramMatching(reference)");
    std::vector<cv::Mat> sourceChannels, referenceChannels, outputChannels;
    cv::split(image, sourceChannels);
    cv::split(reference, referenceChannels);
    for (int c = 0; c < 3; ++c) {
        int sourceHist[256] = {}, referenceHist[256] = {};
        for (int y = 0; y < sourceChannels[c].rows; ++y)
            for (int x = 0; x < sourceChannels[c].cols; ++x)
                sourceHist[sourceChannels[c].at<unsigned char>(y, x)]++;
        for (int y = 0; y < referenceChannels[c].rows; ++y)
            for (int x = 0; x < referenceChannels[c].cols; ++x)
                referenceHist[referenceChannels[c].at<unsigned char>(y, x)]++;
        double sourceCdf[256], referenceCdf[256];
        double totalSource = sourceChannels[c].total();
        double totalReference = referenceChannels[c].total();
        double sumSource = 0.0, sumReference = 0.0;
        for (int i = 0; i < 256; ++i) {
            sumSource += sourceHist[i] / totalSource;
            sumReference += referenceHist[i] / totalReference;
            sourceCdf[i] = sumSource;
            referenceCdf[i] = sumReference;
        }
        unsigned char lookup[256];
        for (int i = 0; i < 256; ++i) {
            int best = 0;
            double difference = 1.0;
            for (int j = 0; j < 256; ++j) {
                double current = std::abs(sourceCdf[i] - referenceCdf[j]);
                if (current < difference) {
                    difference = current;
                    best = j;
                }
            }
            lookup[i] = static_cast<unsigned char>(best);
        }
        cv::Mat matched(sourceChannels[c].size(), CV_8UC1);
        for (int y = 0; y < matched.rows; ++y)
            for (int x = 0; x < matched.cols; ++x)
                matched.at<unsigned char>(y, x) =
                    lookup[sourceChannels[c].at<unsigned char>(y, x)];
        outputChannels.push_back(matched);
    }
    cv::Mat output;
    cv::merge(outputChannels, output);
    return output;
}
cv::Mat affineTransform(const cv::Mat& image, double angleDegrees,
                        double scale, double translateX, double translateY)
{
    validateImage(image, "affineTransform()");
    if (scale <= 0.0) throw std::invalid_argument("affineTransform(): scale must be positive");
    cv::Mat matrix = cv::getRotationMatrix2D(
        cv::Point2f(image.cols / 2.0F, image.rows / 2.0F), angleDegrees, scale);
    matrix.at<double>(0, 2) += translateX;
    matrix.at<double>(1, 2) += translateY;
    cv::Mat output;
    cv::warpAffine(image, output, matrix, image.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar());
    return output;
}

cv::Mat perspectiveTransform(const cv::Mat& image,
    const std::vector<cv::Point2f>& sourcePoints,
    const std::vector<cv::Point2f>& destinationPoints)
{
    validateImage(image, "perspectiveTransform()");
    if (sourcePoints.size() != 4 || destinationPoints.size() != 4)
        throw std::invalid_argument("perspectiveTransform(): exactly four point pairs are required");
    cv::Mat matrix = cv::getPerspectiveTransform(sourcePoints, destinationPoints);
    cv::Mat output;
    cv::warpPerspective(image, output, matrix, image.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar());
    return output;
}

cv::Mat logPolarTransform(const cv::Mat& image, double magnitude)
{
    validateImage(image, "logPolarTransform()");
    if (magnitude <= 0.0) throw std::invalid_argument("logPolarTransform(): magnitude must be positive");
    cv::Mat output;
    cv::logPolar(image, output,
        cv::Point2f(image.cols / 2.0F, image.rows / 2.0F),
        magnitude, cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
    return output;
}

cv::Mat spatialFilter(const cv::Mat& image, int preset, int kernelSize)
{
    validateImage(image, "spatialFilter()");
    if (kernelSize < 3 || kernelSize > 31 || kernelSize % 2 == 0)
        throw std::invalid_argument("spatialFilter(): kernel size must be odd and from 3 to 31");
    cv::Mat output;
    switch (preset) {
    case 0: cv::blur(image, output, cv::Size(kernelSize, kernelSize)); break;
    case 1: cv::GaussianBlur(image, output, cv::Size(kernelSize, kernelSize), 0.0); break;
    case 2: cv::medianBlur(image, output, kernelSize); break;
    case 3: cv::bilateralFilter(image, output, kernelSize, 75.0, 75.0); break;
    case 4: {
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            0, -1, 0, -1, 5, -1, 0, -1, 0);
        cv::filter2D(image, output, -1, kernel);
        break;
    }
    default: throw std::invalid_argument("spatialFilter(): preset must be 0 through 4");
    }
    return output;
}

cv::Mat customSpatialFilter(const cv::Mat& image, const cv::Mat& kernel)
{
    validateImage(image, "customSpatialFilter()");
    if (kernel.empty() || kernel.channels() != 1 ||
        kernel.rows % 2 == 0 || kernel.cols % 2 == 0 ||
        kernel.rows > 15 || kernel.cols > 15)
        throw std::invalid_argument("customSpatialFilter(): kernel must be nonempty, single-channel, odd-sized, and at most 15x15");
    cv::Mat floatKernel;
    kernel.convertTo(floatKernel, CV_32F);
    cv::Mat output;
    cv::filter2D(image, output, -1, floatKernel);
    return output;
}

cv::Mat frequencyFilter(const cv::Mat& image, bool lowPass, int shape,
                        double cutoff, int butterworthOrder)
{
    validateImage(image, "frequencyFilter()");
    if (shape < 0 || shape > 2 || cutoff <= 0.0 ||
        cutoff > std::hypot(image.cols, image.rows) / 2.0 ||
        butterworthOrder < 1 || butterworthOrder > 8)
        throw std::invalid_argument("frequencyFilter(): invalid filter parameters");

    cv::Mat gray, floatImage;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(floatImage, CV_32F);
    cv::Mat planes[] = {floatImage, cv::Mat::zeros(floatImage.size(), CV_32F)};
    cv::Mat spectrum;
    cv::merge(planes, 2, spectrum);
    cv::dft(spectrum, spectrum);
    cv::Mat shifted = spectrum.clone();
    const int cx = shifted.cols / 2, cy = shifted.rows / 2;
    for (int y = 0; y < shifted.rows; ++y)
        for (int x = 0; x < shifted.cols; ++x) {
            const int sx = (x + cx) % shifted.cols;
            const int sy = (y + cy) % shifted.rows;
            shifted.at<cv::Vec2f>(y, x) = spectrum.at<cv::Vec2f>(sy, sx);
        }
    for (int y = 0; y < shifted.rows; ++y)
        for (int x = 0; x < shifted.cols; ++x) {
            const double distance = std::hypot(x - cx, y - cy);
            double low = 0.0;
            if (shape == 0) low = distance <= cutoff ? 1.0 : 0.0;
            else if (shape == 1)
                low = 1.0 / (1.0 + std::pow(distance / cutoff, 2.0 * butterworthOrder));
            else low = std::exp(-(distance * distance) / (2.0 * cutoff * cutoff));
            const float gain = static_cast<float>(lowPass ? low : 1.0 - low);
            shifted.at<cv::Vec2f>(y, x) *= gain;
        }
    cv::Mat unshifted = shifted.clone();
    for (int y = 0; y < shifted.rows; ++y)
        for (int x = 0; x < shifted.cols; ++x) {
            const int sx = (x + cx) % shifted.cols;
            const int sy = (y + cy) % shifted.rows;
            unshifted.at<cv::Vec2f>(sy, sx) = shifted.at<cv::Vec2f>(y, x);
        }
    cv::Mat restored;
    cv::idft(unshifted, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    cv::Mat output;
    cv::normalize(restored, restored, 0, 255, cv::NORM_MINMAX);
    restored.convertTo(restored, CV_8U);
    cv::cvtColor(restored, output, cv::COLOR_GRAY2BGR);
    return output;
}

cv::Mat imageFeatureVector(const cv::Mat& image)
{
    validateImage(image, "imageFeatureVector()");
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    int sizes[] = {8, 4, 4};
    int channelsUsed[] = {0, 1, 2};
    float hRanges[] = {0, 180}, sRanges[] = {0, 256}, vRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges, vRanges};
    cv::Mat histogram;
    cv::calcHist(&hsv, 1, channelsUsed, cv::Mat(), histogram, 3, sizes, ranges, true, false);
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram.reshape(1, 1).clone();
}

namespace {
ClassificationResult classifyFeature(const cv::Mat& feature,
    const std::vector<cv::Mat>& classSamples, const std::vector<int>& labels,
    bool cosine)
{
    if (feature.empty() || classSamples.empty() ||
        classSamples.size() != labels.size())
        throw std::invalid_argument("classifier: samples and labels must be nonempty and match");

    cv::Mat query;
    feature.convertTo(query, CV_32F);
    query = query.reshape(1, 1);

    std::vector<int> uniqueLabels;
    std::vector<cv::Mat> means;
    std::vector<int> counts;
    for (size_t i = 0; i < classSamples.size(); ++i) {
        cv::Mat sample;
        classSamples[i].convertTo(sample, CV_32F);
        sample = sample.reshape(1, 1);
        if (sample.total() != query.total())
            throw std::invalid_argument("classifier: feature dimensions do not match");
        auto found = std::find(uniqueLabels.begin(), uniqueLabels.end(), labels[i]);
        if (found == uniqueLabels.end()) {
            uniqueLabels.push_back(labels[i]);
            means.push_back(sample.clone());
            counts.push_back(1);
        } else {
            const size_t index = static_cast<size_t>(found - uniqueLabels.begin());
            means[index] += sample;
            ++counts[index];
        }
    }

    ClassificationResult result;
    double best = cosine ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < means.size(); ++i) {
        means[i] /= static_cast<float>(counts[i]);
        double score;
        if (cosine) {
            const double denominator = cv::norm(query) * cv::norm(means[i]);
            score = denominator > 0.0 ? query.dot(means[i]) / denominator : 0.0;
            if (score > best) { best = score; result.label = uniqueLabels[i]; }
        } else {
            score = cv::norm(query, means[i], cv::NORM_L2);
            if (score < best) { best = score; result.label = uniqueLabels[i]; }
        }
    }
    result.score = best;
    return result;
}
}

ClassificationResult minimumDistanceClassifier(
    const cv::Mat& feature, const std::vector<cv::Mat>& classSamples,
    const std::vector<int>& labels)
{ return classifyFeature(feature, classSamples, labels, false); }

ClassificationResult cosineSimilarityClassifier(
    const cv::Mat& feature, const std::vector<cv::Mat>& classSamples,
    const std::vector<int>& labels)
{ return classifyFeature(feature, classSamples, labels, true); }

double classificationAccuracy(
    const std::vector<cv::Mat>& samples,
    const std::vector<int>& labels,
    bool useCosineSimilarity)
{
    if (samples.size() != labels.size() || samples.size() < 4)
        return -1.0;
    int correct = 0;
    int tested = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        std::vector<cv::Mat> training;
        std::vector<int> trainingLabels;
        for (size_t j = 0; j < samples.size(); ++j) {
            if (j != i) {
                training.push_back(samples[j]);
                trainingLabels.push_back(labels[j]);
            }
        }
        if (std::find(trainingLabels.begin(), trainingLabels.end(), labels[i]) ==
            trainingLabels.end())
            continue;
        const ClassificationResult result = useCosineSimilarity
            ? cosineSimilarityClassifier(samples[i], training, trainingLabels)
            : minimumDistanceClassifier(samples[i], training, trainingLabels);
        if (result.label == labels[i]) ++correct;
        ++tested;
    }
    return tested > 0 ? 100.0 * correct / tested : -1.0;
}

}  // namespace photo_booth
