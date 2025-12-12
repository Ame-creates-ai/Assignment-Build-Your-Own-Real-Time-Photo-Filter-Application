# Real-Time Photo Filter Application

A Python-based real-time photo filter application with webcam capture and creative filters, including a custom Peanut Filter that overlays a peanut-shaped mask over detected faces.

## Features

### Standard Filters
- **Box Blur**: 5×5 and 11×11 kernel options (toggle with 'B')
- **Gaussian Blur**: Adjustable sigma ('+'/'-' keys to adjust)
- **Sharpening**: Kernel-based sharpening filter
- **Sobel Edge Detection**: Gradient-based edge detection
- **Canny Edge Detection**: Advanced edge detection with adjustable thresholds ('+'/'-' keys)

### Creative Filter
- **Peanut Filter**: Custom face detection filter that overlays a peanut-shaped mask over faces while keeping eyes and mouth visible

### UI Features
- Real-time FPS counter
- Current filter name display
- Help text with all keyboard shortcuts (toggle with 'H')
- Image saving capability (SPACE key)
- Performance optimizations for 15+ FPS

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python photo_filter_app.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **B** | Box Blur (toggles 5×5 and 11×11) |
| **G** | Gaussian Blur |
| **S** | Sharpening |
| **E** | Sobel Edge Detection |
| **C** | Canny Edge Detection |
| **A** | Peanut Filter |
| **N** | No Filter |
| **H** | Toggle Help Text |
| **+/-** | Adjust filter parameters (Gaussian sigma, Canny thresholds) |
| **SPACE** | Save Current Frame |
| **Q** | Quit Application |

## Technical Details

- **Language**: Python 3
- **Libraries**: OpenCV (cv2), NumPy
- **Face Detection**: Haar Cascade Classifier
- **Performance**: Optimized for 15+ FPS with frame skipping for face detection
- **Output**: Filtered images saved to `filtered_images/` directory

## Peanut Filter Details

The Peanut Filter:
1. Detects faces using Haar Cascade
2. Creates an elliptical peanut-shaped overlay in tan/brown color
3. Estimates eye positions (upper third of face) and creates circular cutouts
4. Estimates mouth position (lower third of face) and creates a circular cutout
5. Blends the peanut overlay with the original face while preserving eye and mouth regions
6. Tracks faces across frames for performance optimization

## Performance Optimization

- Face detection runs every 5 frames, with tracking in between frames
- Frame resizing for faster processing
- Real-time FPS monitoring
- Efficient NumPy operations for image blending

## Troubleshooting

- **Webcam not detected**: Ensure your webcam is connected and not in use by another application
- **Low FPS**: Reduce frame resolution or increase face detection interval
- **Peanut filter not detecting faces**: Ensure adequate lighting and face is clearly visible

