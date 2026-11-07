#include "photo_booth/AppConfig.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>

namespace photo_booth {

namespace {

void validateKeys(const toml::table& table,
                  const std::set<std::string>& valid_keys,
                  const std::string& section) {
  for (const auto& [key, value] : table) {
    (void)value;
    const std::string key_string{key.str()};

    if (valid_keys.find(key_string) == valid_keys.end()) {
      throw std::runtime_error("Unknown configuration option: " + section +
                               "." + key_string);
    }
  }
}

void validateConfigKeys(const toml::table& table) {
  const std::set<std::string> valid_sections{
      "camera",
      "preview",
      "capture",
      "processing",
      "corner_detection",
      "classification",
  };

  for (const auto& [key, value] : table) {
    const std::string section{key.str()};

    if (valid_sections.find(section) == valid_sections.end()) {
      throw std::runtime_error("Unknown configuration section: " + section);
    }

    if (!value.is_table()) {
      throw std::runtime_error("Configuration section '" + section +
                               "' must be a TOML table.");
    }
  }

  if (const auto* camera = table["camera"].as_table()) {
    validateKeys(*camera, {"device", "width", "height", "fps", "fourcc"},
                 "camera");
  }

  if (const auto* preview = table["preview"].as_table()) {
    validateKeys(*preview, {"mirror", "rotation", "window_name"}, "preview");
  }

  if (const auto* capture = table["capture"].as_table()) {
    validateKeys(*capture, {"output_filename", "warmup_frames"}, "capture");
  }

  if (const auto* processing = table["processing"].as_table()) {
    validateKeys(*processing, {"channel_swap_enabled"}, "processing");
  }

  if (const auto* corners = table["corner_detection"].as_table()) {
    validateKeys(
        *corners,
        {"enabled", "method", "harris_block_size",
         "harris_aperture_size", "harris_k",
         "harris_threshold_ratio", "fast_threshold",
         "fast_nonmax_suppression"},
        "corner_detection");
  }

  if (const auto* classification = table["classification"].as_table()) {
    validateKeys(
        *classification,
        {"enabled", "method", "reference_directory"},
        "classification");
  }
}

template <typename T>
void readOptional(const toml::table& table,
                  const std::string& section,
                  const std::string& key,
                  T& destination) {
  const auto node = table[section][key];

  if (!node) {
    return;
  }

  const auto value = node.template value<T>();

  if (!value) {
    throw std::runtime_error("Configuration option " + section + "." + key +
                             " has the wrong type.");
  }

  destination = *value;
}

void validateValues(const AppConfig& config) {
  if (config.camera.device < 0) {
    throw std::runtime_error("camera.device must be >= 0.");
  }

  if (config.camera.width < 0) {
    throw std::runtime_error("camera.width must be >= 0.");
  }

  if (config.camera.height < 0) {
    throw std::runtime_error("camera.height must be >= 0.");
  }

  if (!std::isfinite(config.camera.fps) || config.camera.fps < 0.0) {
    throw std::runtime_error("camera.fps must be finite and >= 0.");
  }

  if (!config.camera.fourcc.empty() && config.camera.fourcc.size() != 4) {
    throw std::runtime_error(
        "camera.fourcc must be empty or contain exactly four characters.");
  }

  if (config.preview.rotation != 0 && config.preview.rotation != 90 &&
      config.preview.rotation != 180 && config.preview.rotation != 270) {
    throw std::runtime_error("preview.rotation must be 0, 90, 180, or 270.");
  }

  if (config.preview.window_name.empty()) {
    throw std::runtime_error("preview.window_name cannot be empty.");
  }

  if (config.capture.output_filename.empty()) {
    throw std::runtime_error("capture.output_filename cannot be empty.");
  }

  if (config.capture.warmup_frames < 0) {
    throw std::runtime_error("capture.warmup_frames must be >= 0.");
  }

  if (config.corner_detection.method != "harris" &&
      config.corner_detection.method != "fast") {
    throw std::runtime_error(
        "corner_detection.method must be 'harris' or 'fast'.");
  }

  if (config.corner_detection.harris_block_size <= 0) {
    throw std::runtime_error(
        "corner_detection.harris_block_size must be positive.");
  }

  if (config.corner_detection.harris_aperture_size < 3 ||
      config.corner_detection.harris_aperture_size > 7 ||
      config.corner_detection.harris_aperture_size % 2 == 0) {
    throw std::runtime_error(
        "corner_detection.harris_aperture_size must be 3, 5, or 7.");
  }

  if (!std::isfinite(config.corner_detection.harris_k) ||
      config.corner_detection.harris_k <= 0.0) {
    throw std::runtime_error(
        "corner_detection.harris_k must be finite and positive.");
  }

  if (!std::isfinite(config.corner_detection.harris_threshold_ratio) ||
      config.corner_detection.harris_threshold_ratio < 0.0 ||
      config.corner_detection.harris_threshold_ratio > 1.0) {
    throw std::runtime_error(
        "corner_detection.harris_threshold_ratio must be from 0 to 1.");
  }

  if (config.corner_detection.fast_threshold <= 0) {
    throw std::runtime_error(
        "corner_detection.fast_threshold must be positive.");
  }

  if (config.classification.method != "minimum_distance" &&
      config.classification.method != "cosine") {
    throw std::runtime_error(
        "classification.method must be 'minimum_distance' or 'cosine'.");
  }

  if (config.classification.reference_directory.empty()) {
    throw std::runtime_error(
        "classification.reference_directory cannot be empty.");
  }
}

}  // namespace

AppConfig loadConfig(const std::filesystem::path& filename) {
  AppConfig config;
  toml::table table;

  try {
    table = toml::parse_file(filename.string());
  } catch (const toml::parse_error& error) {
    throw std::runtime_error("Error parsing configuration file '" +
                             filename.string() +
                             "': " + std::string{error.description()});
  }

  validateConfigKeys(table);

  readOptional(table, "camera", "device", config.camera.device);
  readOptional(table, "camera", "width", config.camera.width);
  readOptional(table, "camera", "height", config.camera.height);
  readOptional(table, "camera", "fps", config.camera.fps);
  readOptional(table, "camera", "fourcc", config.camera.fourcc);

  readOptional(table, "preview", "mirror", config.preview.mirror);
  readOptional(table, "preview", "rotation", config.preview.rotation);
  readOptional(table, "preview", "window_name", config.preview.window_name);

  readOptional(table, "capture", "output_filename",
               config.capture.output_filename);
  readOptional(table, "capture", "warmup_frames",
               config.capture.warmup_frames);

  readOptional(table, "processing", "channel_swap_enabled",
               config.processing.channel_swap_enabled);

  readOptional(table, "corner_detection", "enabled",
               config.corner_detection.enabled);
  readOptional(table, "corner_detection", "method",
               config.corner_detection.method);
  readOptional(table, "corner_detection", "harris_block_size",
               config.corner_detection.harris_block_size);
  readOptional(table, "corner_detection", "harris_aperture_size",
               config.corner_detection.harris_aperture_size);
  readOptional(table, "corner_detection", "harris_k",
               config.corner_detection.harris_k);
  readOptional(table, "corner_detection", "harris_threshold_ratio",
               config.corner_detection.harris_threshold_ratio);
  readOptional(table, "corner_detection", "fast_threshold",
               config.corner_detection.fast_threshold);
  readOptional(table, "corner_detection", "fast_nonmax_suppression",
               config.corner_detection.fast_nonmax_suppression);

  readOptional(table, "classification", "enabled",
               config.classification.enabled);
  readOptional(table, "classification", "method",
               config.classification.method);

  std::string reference_directory =
      config.classification.reference_directory.string();

  readOptional(table, "classification", "reference_directory",
               reference_directory);

  config.classification.reference_directory = reference_directory;

  validateValues(config);

  return config;
}

ImageCapture::Configuration makeImageCaptureConfiguration(
    const CameraConfig& camera_config) {
  ImageCapture::Configuration configuration;

  configuration.device_index = camera_config.device;
  configuration.width = camera_config.width;
  configuration.height = camera_config.height;
  configuration.frames_per_second = camera_config.fps;
  configuration.fourcc = camera_config.fourcc;

  return configuration;
}

}  // namespace photo_booth
