# Photo Booth

A C++20/OpenCV photo booth application for IMGS.361 Image Processing.

The project keeps the original semester-project architecture: camera acquisition and configuration remain separate from the reusable image-processing functions, while `apps/photo_booth.cpp` owns the interactive runtime state and processing order.

## Final processing capabilities

The photo booth implements eight substantive image-processing capabilities:

1. **Channel swapping** — swaps the red and blue channels.
2. **Dynamic contrast enhancement** — stretches each channel using percentile-based low/high values.
3. **Grayscale conversion** — converts the image to grayscale while preserving the project's three-channel BGR representation.
4. **Gaussian blur** — applies a configurable odd-sized Gaussian spatial filter.
5. **Canny edge detection** — detects image edges and converts the result back to three-channel BGR.
6. **Image inversion** — computes the photographic negative of the image.
7. **Image quantization** — reduces each channel to a selectable number of intensity levels.
8. **Arbitrary-angle rotation** — rotates the image using inverse coordinate mapping.

The application also provides a working photo-capture workflow with a three-second countdown, processed-image capture, and unique output filenames.

## Processing pipeline

The active operations are applied in this order:

```text
camera
  ↓
channel swap
  ↓
dynamic contrast
  ↓
grayscale
  ↓
Gaussian blur
  ↓
Canny edge detection
  ↓
inversion
  ↓
quantization
  ↓
rotation
  ↓
preview / capture
```

Operations controlled by the keyboard can be enabled independently, so multiple capabilities can be combined in a single processing pipeline.

## Image representation

Successful camera frames use OpenCV's `CV_8UC3` BGR representation.

Processing functions preserve that representation. In particular, grayscale and edge detection internally use a single-channel image but convert their results back to BGR before returning. This allows the processing operations to be composed without changing the pipeline's image type.

## Controls

Run the application and use the following keys while the preview window has focus:

| Key | Function |
| --- | --- |
| `C` | Toggle dynamic contrast |
| `I` | Toggle inversion |
| `N` | Toggle quantization |
| `G` | Toggle grayscale |
| `B` | Toggle Gaussian blur |
| `E` | Toggle Canny edge detection |
| `R` | Toggle rotation |
| `[` / `]` | Decrease / increase quantization levels |
| `-` / `+` | Decrease / increase blur kernel size |
| `,` / `.` | Decrease / increase rotation angle |
| `SPACE` | Start a 3-second capture countdown |
| `Q` or `Esc` | Quit |

Quantization levels range from 2 through 256. Blur kernel sizes range from 3 through 31 and remain odd. Rotation ranges from -180 through +180 degrees in five-degree steps.

## Capture

Press `SPACE` to capture the currently processed image.

The application:

1. Freezes the current processed frame.
2. Displays a three-second countdown.
3. Saves the processed image.
4. Automatically chooses the next unused filename.

With the default configuration, captures are saved as:

```text
captured_image_001.png
captured_image_002.png
captured_image_003.png
```

If a directory is included in `capture.output_filename`, the directory is created automatically when needed.

## Project organization

```text
Photo-Booth-Final/
├── CMakeLists.txt
├── config.toml
├── LICENSE
├── README.md
├── apps/
│   ├── capture_single_image.cpp
│   ├── live_preview.cpp
│   └── photo_booth.cpp
├── include/
│   └── photo_booth/
│       ├── AppConfig.hpp
│       ├── ImageCapture.hpp
│       └── ImageProcessing.hpp
├── src/
│   ├── AppConfig.cpp
│   ├── ImageCapture.cpp
│   └── ImageProcessing.cpp
└── tools/
```

The Week 15 work is intentionally concentrated in:

```text
apps/photo_booth.cpp
include/photo_booth/ImageProcessing.hpp
src/ImageProcessing.cpp
config.toml
README.md
```

No new framework or processing class hierarchy is required. Processing algorithms remain ordinary functions and `processFrame()` continues to determine the pipeline order.

## Requirements

- CMake 3.30 or later
- C++20 compiler
- OpenCV 4.x or 5.x with `core`, `videoio`, `highgui`, `imgcodecs`, and `imgproc`
- Boost with the `program_options` component
- Eigen3
- toml++ 3.4.0

The existing CMake configuration downloads toml++ with `FetchContent`.

## Build from a clean clone

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

The executables are placed in:

```text
build/bin/
├── capture_single_image
├── live_preview
└── photo_booth
```

For a clean-clone verification, remove the entire `build` directory before running the commands above.

## Run the photo booth

```bash
./build/bin/photo_booth
```

Or provide a different TOML configuration:

```bash
./build/bin/photo_booth webcam.toml
```

The default configuration is `config.toml` in the current working directory.

## Other applications

Minimal live preview:

```bash
./build/bin/live_preview
```

Single-image diagnostic utility:

```bash
./build/bin/capture_single_image
```

All applications also support:

```bash
./build/bin/photo_booth --help
```

## Configuration

The supplied `config.toml` contains camera, preview, capture, and processing settings.

The processing effects added for the final application are intentionally runtime-controlled rather than adding a large configuration system. This keeps the final architecture close to the semester's existing design while allowing the user to combine processing capabilities interactively.

## Design notes

The application uses an explicit processing pipeline rather than a general-purpose processing framework.

`ImageProcessing.cpp` contains reusable image-processing algorithms.

`photo_booth.cpp` contains:

- `ProcessingState` for runtime effect state.
- `processFrame()` for pipeline order.
- `showPreviewFrame()` for the interactive display.
- `handleKey()` for keyboard controls.
- Capture/countdown behavior.

This preserves the architec
