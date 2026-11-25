# Photo Booth

A C++20/OpenCV photo booth application for IMGS.361 Image Processing.

The project preserves the existing semester-project architecture. Camera acquisition and configuration remain separate from reusable image-processing functions, while `apps/photo_booth.cpp` manages runtime state, keyboard controls, and processing order.

## Processing capabilities

The project includes the original image-processing features and the additional grouped capabilities:

1. **Contrast enhancement**
   - Dynamic percentile contrast
   - Linear contrast stretching
   - Histogram equalization
   - Histogram matching using a captured reference frame

2. **Geometric transformations**
   - Rotation
   - Affine transformation
   - Perspective transformation
   - Log-polar transformation

3. **Spatial filtering**
   - Box averaging
   - Gaussian filtering
   - Median filtering
   - Bilateral filtering
   - Sharpening
   - User-entered custom convolution kernel

4. **Frequency-domain filtering**
   - Ideal low-pass and high-pass filters
   - Butterworth low-pass and high-pass filters
   - Gaussian low-pass and high-pass filters

5. **Corner detection**
   - Harris corner detection
   - FAST corner detection

6. **Data embedding**
   - Embeds a built-in camera icon into selected Fourier phase coefficients in the luminance channel.

7. **Classification**
   - Minimum-distance-to-class-mean classification
   - Cosine-similarity classification
   - Leave-one-out classification accuracy reporting

Other included features are channel swapping, grayscale conversion, Gaussian blur, Canny edge detection, image inversion, and image quantization.

## Processing pipeline

The application processes each camera frame through the enabled operations:

```text
camera frame
  ↓
channel swapping
  ↓
contrast enhancement
  ↓
grayscale conversion
  ↓
spatial filtering
  ↓
frequency filtering
  ↓
edge detection
  ↓
inversion
  ↓
quantization
  ↓
geometric transformation
  ↓
Harris or FAST corner detection
  ↓
Fourier phase data embedding
  ↓
preview / capture
- `showPreviewFrame()` for the interactive display.
- `handleKey()` for keyboard controls.
- Capture/countdown behavior.

This preserves the architec
