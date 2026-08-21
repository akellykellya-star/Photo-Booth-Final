#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "photo_booth/AppConfig.hpp"
#include "photo_booth/ImageCapture.hpp"
#include "photo_booth/ImageProcessing.hpp"

namespace {
struct ProcessingState {
    bool inversion_enabled{false};
    bool quantization_enabled{false};
    int quantization_levels{8};
    bool rotation_enabled{false};
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

    processed_frame =
        photo_booth::dynamicContrast(
            processed_frame,
            1,
            99);

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

void showPreviewFrame(
    const cv::Mat& frame,
    const photo_booth::PreviewConfig& config,
    const ProcessingState& state)
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

    /*
     * Display the available Week 5 controls.
     */
    cv::putText(
        preview_frame,
        "N: Quantization  R: Rotation  I: Invert  Q: Quit",
        cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1);

    /*
     * Display the current processing state.
     */
    std::string status =
        "Quantization: " +
        std::string(
            state.quantization_enabled
                ? "ON"
                : "OFF");

    status +=
        "   Rotation: " +
        std::string(
            state.rotation_enabled
                ? "ON"
                : "OFF");

    status +=
        "   Invert: " +
        std::string(
            state.inversion_enabled
                ? "ON"
                : "OFF");

    cv::putText(
        preview_frame,
        status,
        cv::Point(10, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1);

    cv::imshow(
        config.window_name,
        preview_frame);
}

bool handleKey(
    const int key,
    ProcessingState& state)
{
    switch (key) {
    case 27:
    case 'q':
    case 'Q':
        return false;
    case 'i':
    case 'I':
        state.inversion_enabled =
            !state.inversion_enabled;
        std::cout
            << "Image inversion: "
            << (state.inversion_enabled
                    ? "ON"
                    : "OFF")
            << '\n';

        break;
    case 'n':
    case 'N':
        state.quantization_enabled =
            !state.quantization_enabled;
        std::cout
            << "Quantization: "
            << (state.quantization_enabled
                    ? "ON"
                    : "OFF")
            << " ("
            << state.quantization_levels
            << " levels)"
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

    case 'r':
    case 'R':

        state.rotation_enabled =
            !state.rotation_enabled;

        std::cout
            << "Rotation: "
            << (state.rotation_enabled
                    ? "ON"
                    : "OFF")
            << " ("
            << state.rotation_angle
            << " degrees)"
            << '\n';

        break;

    case '<':

        state.rotation_angle -= 5.0;

        if (state.rotation_angle < -180.0) {
            state.rotation_angle = -180.0;
        }

        std::cout
            << "Rotation angle: "
            << state.rotation_angle
            << " degrees"
            << '\n';

        break;

    case '>':

        state.rotation_angle += 5.0;

        if (state.rotation_angle > 180.0) {
            state.rotation_angle = 180.0;
        }

        std::cout
            << "Rotation angle: "
            << state.rotation_angle
            << " degrees"
            << '\n';

        break;
    default:
        break;
    }

    return true;
}

}


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
                    << "Runs the semester photo-booth "
                    << "image-processing application.\n";
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
            cv::waitKeyEx(1);

        if (key != -1) {
            std::cout
            << "Key received: "
            << key
            << '\n';
    }

        if (!handleKey(
            key,
            processing_state)) {
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
