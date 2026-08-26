#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "photo_booth/AppConfig.hpp"
#include "photo_booth/ImageCapture.hpp"
#include "photo_booth/ImageProcessing.hpp"

namespace {

struct ProcessingState {
    bool contrast_enabled{true};
    bool inversion_enabled{false};
    bool quantization_enabled{false};
    bool rotation_enabled{false};

    // Week 15 additions.
    bool grayscale_enabled{false};
    bool blur_enabled{false};
    bool edge_enabled{false};

    int quantization_levels{8};
    int blur_kernel_size{5};

    double rotation_angle{0.0};
};

cv::Mat processFrame(
    const cv::Mat& frame,
    const photo_booth::ProcessingConfig& config,
    const ProcessingState& state)
{
    cv::Mat processed_frame =
        frame.clone();

    if (config.channel_swap_enabled) {
        processed_frame =
            photo_booth::swapRedBlueChannels(
                processed_frame);
    }

    if (state.contrast_enabled) {
        processed_frame =
            photo_booth::dynamicContrast(
                processed_frame,
                1,
                99);
    }

    // New spatial-processing stages.
    if (state.grayscale_enabled) {
        processed_frame =
            photo_booth::grayscaleImage(
                processed_frame);
    }

    if (state.blur_enabled) {
        processed_frame =
            photo_booth::blurImage(
                processed_frame,
                state.blur_kernel_size);
    }

    if (state.edge_enabled) {
        processed_frame =
            photo_booth::edgeDetectImage(
                processed_frame,
                50.0,
                150.0);
    }

    if (state.inversion_enabled) {
        processed_frame =
            photo_booth::invertImage(
                processed_frame);
    }

    if (state.quantization_enabled) {
        processed_frame =
            photo_booth::quantizeImage(
                processed_frame,
                state.quantization_levels);
    }

    if (state.rotation_enabled) {
        processed_frame =
            photo_booth::rotateImage(
                processed_frame,
                state.rotation_angle);
    }

    return processed_frame;
}

void drawText(
    cv::Mat& image,
    const std::string& text,
    int x,
    int y,
    double scale = 0.45)
{
    cv::putText(
        image,
        text,
        cv::Point(x, y),
        cv::FONT_HERSHEY_SIMPLEX,
        scale,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);
}

void showPreviewFrame(
    const cv::Mat& frame,
    const photo_booth::PreviewConfig& config,
    const ProcessingState& state,
    const std::string& capture_message = {})
{
    cv::Mat preview_frame =
        frame.clone();

    switch (config.rotation) {
    case 0:
        break;

    case 90:
        cv::rotate(
            preview_frame,
            preview_frame,
            cv::ROTATE_90_CLOCKWISE);
        break;

    case 180:
        cv::rotate(
            preview_frame,
            preview_frame,
            cv::ROTATE_180);
        break;

    case 270:
        cv::rotate(
            preview_frame,
            preview_frame,
            cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
    }

    if (config.mirror) {
        cv::flip(
            preview_frame,
            preview_frame,
            1);
    }

    drawText(
        preview_frame,
        "PHOTO BOOTH",
        10,
        25,
        0.65);

    std::string status =
        "C Contrast " +
        std::string(
            state.contrast_enabled ? "ON" : "OFF") +
        "   I Invert " +
        std::string(
            state.inversion_enabled ? "ON" : "OFF") +
        "   N Quantize " +
        std::string(
            state.quantization_enabled ? "ON" : "OFF");

    drawText(
        preview_frame,
        status,
        10,
        50);

    std::string spatial =
        "G Grayscale " +
        std::string(
            state.grayscale_enabled ? "ON" : "OFF") +
        "   B Blur " +
        std::string(
            state.blur_enabled ? "ON" : "OFF") +
        "   E Edges " +
        std::string(
            state.edge_enabled ? "ON" : "OFF");

    drawText(
        preview_frame,
        spatial,
        10,
        74);

    std::string parameters =
        "R Rotate " +
        std::string(
            state.rotation_enabled ? "ON" : "OFF") +
        " (" +
        std::to_string(
            static_cast<int>(state.rotation_angle)) +
        " deg)";

    parameters +=
        "   Levels: " +
        std::to_string(
            state.quantization_levels);

    parameters +=
        "   Blur: " +
        std::to_string(
            state.blur_kernel_size);

    drawText(
        preview_frame,
        parameters,
        10,
        98);

    drawText(
        preview_frame,
        "C I N G B E R",
        10,
        126);

    drawText(
        preview_frame,
        "[ ] Levels   - + Blur   , . Angle   SPACE Capture   Q Quit",
        10,
        149);

    if (!capture_message.empty()) {
        drawText(
            preview_frame,
            capture_message,
            10,
            preview_frame.rows - 20,
            0.55);
    }

    cv::imshow(
        config.window_name,
        preview_frame);
}

std::filesystem::path nextCaptureFilename(
    const std::string& configured_filename)
{
    std::filesystem::path configured{
        configured_filename};

    std::filesystem::path directory =
        configured.parent_path();

    if (!directory.empty()) {
        std::filesystem::create_directories(
            directory);
    }

    const std::string stem =
        configured.stem().string().empty()
            ? "captured_image"
            : configured.stem().string();

    std::string extension =
        configured.extension().string();

    if (extension.empty()) {
        extension = ".png";
    }

    for (int index = 1; index <= 9999; ++index) {
        std::ostringstream filename;
        filename
            << stem
            << "_"
            << std::setfill('0')
            << std::setw(3)
            << index
            << extension;

        const auto candidate =
            directory / filename.str();

        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    throw std::runtime_error(
        "Unable to find an available capture filename.");
}

bool captureWithCountdown(
    const cv::Mat& processed_frame,
    const photo_booth::PreviewConfig& preview_config,
    const photo_booth::CaptureConfig& capture_config)
{
    if (processed_frame.empty()) {
        std::cerr
            << "Unable to capture: processed frame is empty.\n";
        return true;
    }

    // Freeze the current processed image during the countdown.
    const cv::Mat capture_frame =
        processed_frame.clone();

    for (int count = 3; count >= 1; --count) {
        cv::Mat countdown_frame =
            capture_frame.clone();

        switch (preview_config.rotation) {
        case 90:
            cv::rotate(
                countdown_frame,
                countdown_frame,
                cv::ROTATE_90_CLOCKWISE);
            break;
        case 180:
            cv::rotate(
                countdown_frame,
                countdown_frame,
                cv::ROTATE_180);
            break;
        case 270:
            cv::rotate(
                countdown_frame,
                countdown_frame,
                cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            break;
        }

        if (preview_config.mirror) {
            cv::flip(
                countdown_frame,
                countdown_frame,
                1);
        }

        drawText(
            countdown_frame,
            std::to_string(count),
            std::max(20, countdown_frame.cols / 2 - 20),
            std::max(60, countdown_frame.rows / 2),
            2.0);

        cv::imshow(
            preview_config.window_name,
            countdown_frame);

        cv::waitKey(1000);
    }

    const auto output_path =
        nextCaptureFilename(
            capture_config.output_filename);

    if (!cv::imwrite(
            output_path.string(),
            capture_frame)) {
        std::cerr
            << "Unable to save image to "
            << output_path.string()
            << '\n';

        return true;
    }

    std::cout
        << "Captured image: "
        << output_path.string()
        << '\n';

    return true;
}

bool handleKey(
    const int key,
    ProcessingState& state,
    const cv::Mat& processed_frame,
    const photo_booth::PreviewConfig& preview_config,
    const photo_booth::CaptureConfig& capture_config)
{
    switch (key) {
    case 27:
    case 'q':
    case 'Q':
        return false;

    case 'c':
    case 'C':
        state.contrast_enabled =
            !state.contrast_enabled;

        std::cout
            << "Dynamic contrast: "
            << (state.contrast_enabled ? "ON" : "OFF")
            << '\n';
        break;

    case 'i':
    case 'I':
        state.inversion_enabled =
            !state.inversion_enabled;

        std::cout
            << "Image inversion: "
            << (state.inversion_enabled ? "ON" : "OFF")
            << '\n';
        break;

    case 'n':
    case 'N':
        state.quantization_enabled =
            !state.quantization_enabled;

        std::cout
            << "Quantization: "
            << (state.quantization_enabled ? "ON" : "OFF")
            << " (" << state.quantization_levels
            << " levels)\n";
        break;

    case 'g':
    case 'G':
        state.grayscale_enabled =
            !state.grayscale_enabled;

        std::cout
            << "Grayscale: "
            << (state.grayscale_enabled ? "ON" : "OFF")
            << '\n';
        break;

    case 'b':
    case 'B':
        state.blur_enabled =
            !state.blur_enabled;

        std::cout
            << "Gaussian blur: "
            << (state.blur_enabled ? "ON" : "OFF")
            << " (" << state.blur_kernel_size
            << "x" << state.blur_kernel_size << ")\n";
        break;

    case 'e':
    case 'E':
        state.edge_enabled =
            !state.edge_enabled;

        std::cout
            << "Edge detection: "
            << (state.edge_enabled ? "ON" : "OFF")
            << '\n';
        break;

    case ']':
        if (state.quantization_levels < 256) {
            state.quantization_levels *= 2;

            if (state.quantization_levels > 256) {
                state.quantization_levels = 256;
            }
        }

        std::cout
            << "Quantization levels: "
            << state.quantization_levels
            << '\n';
        break;

    case '[':
        if (state.quantization_levels > 2) {
            state.quantization_levels /= 2;
        }

        std::cout
            << "Quantization levels: "
            << state.quantization_levels
            << '\n';
        break;

    case '-':
    case '_':
        if (state.blur_kernel_size > 3) {
            state.blur_kernel_size -= 2;
        }

        std::cout
            << "Blur kernel: "
            << state.blur_kernel_size
            << "x" << state.blur_kernel_size
            << '\n';
        break;

    case '+':
    case '=':
        if (state.blur_kernel_size < 31) {
            state.blur_kernel_size += 2;
        }

        std::cout
            << "Blur kernel: "
            << state.blur_kernel_size
            << "x" << state.blur_kernel_size
            << '\n';
        break;

    case 'r':
    case 'R':
        state.rotation_enabled =
            !state.rotation_enabled;

        std::cout
            << "Rotation: "
            << (state.rotation_enabled ? "ON" : "OFF")
            << " (" << state.rotation_angle
            << " degrees)\n";
        break;

    case ',':
        state.rotation_angle -= 5.0;

        if (state.rotation_angle < -180.0) {
            state.rotation_angle = -180.0;
        }

        std::cout
            << "Rotation angle: "
            << state.rotation_angle
            << " degrees\n";
        break;

    case '.':
        state.rotation_angle += 5.0;

        if (state.rotation_angle > 180.0) {
            state.rotation_angle = 180.0;
        }

        std::cout
            << "Rotation angle: "
            << state.rotation_angle
            << " degrees\n";
        break;

    case ' ':
        return captureWithCountdown(
            processed_frame,
            preview_config,
            capture_config);

    default:
        break;
    }

    return true;
}

}  // namespace

int main(
    int argc,
    char* argv[])
{
    try {
        std::filesystem::path config_path{
            "config.toml"};

        if (argc > 2) {
            std::cerr
                << "Usage: "
                << argv[0]
                << " [config.toml]\n";
            return EXIT_FAILURE;
        }

        if (argc == 2) {
            const std::string argument{
                argv[1]};

            if (argument == "-h" ||
                argument == "--help") {
                std::cout
                    << "Usage: "
                    << argv[0]
                    << " [config.toml]\n\n"
                    << "Runs the photo-booth image-processing "
                    << "application.\n";
                return EXIT_SUCCESS;
            }

            config_path = argument;
        }

        const auto config =
            photo_booth::loadConfig(
                config_path);

        photo_booth::ImageCapture camera(
            photo_booth::makeImageCaptureConfiguration(
                config.camera));

        if (!camera.open()) {
            std::cerr
                << camera.errorMessage()
                << '\n';
            return EXIT_FAILURE;
        }

        std::cout
            << camera
            << '\n';

        cv::namedWindow(
            config.preview.window_name,
            cv::WINDOW_AUTOSIZE);

        ProcessingState processing_state;

        while (true) {
            if (!camera.read()) {
                std::cerr
                    << camera.errorMessage()
                    << '\n';
                return EXIT_FAILURE;
            }

            cv::Mat processed_frame =
                processFrame(
                    camera.image(),
                    config.processing,
                    processing_state);

            showPreviewFrame(
                processed_frame,
                config.preview,
                processing_state);

            const int key =
                cv::waitKey(1);

            if (!handleKey(
                    key,
                    processing_state,
                    processed_frame,
                    config.preview,
                    config.capture)) {
                break;
            }
        }

        cv::destroyAllWindows();
    }
    catch (const cv::Exception& error) {
        std::cerr
            << "OpenCV error: "
            << error.what()
            << '\n';
        return EXIT_FAILURE;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
