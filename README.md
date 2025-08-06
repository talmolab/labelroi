# labelroi

A Qt-based application for drawing and labeling polygon/polyline ROIs (Regions of Interest) on images and videos.

## Features

- **Image and Video Support**: Load images (PNG, JPG, JPEG, BMP, TIFF) or MP4 videos (displays first frame)
- **ROI Drawing**: Draw polygon ROIs by clicking vertices, or polylines for open shapes
- **ROI Naming**: Double-click on any ROI to add or edit its name
- **Smooth Navigation**: Pan and zoom with mouse/trackpad gestures
- **ROI Management**: View ROI properties, delete individual ROIs, or clear all
- **YAML Export/Import**: Save ROIs to YAML files for later use or sharing
- **Geometry Analysis**: Automatic calculation of area, perimeter, and vertex count using Shapely

## Installation

### Prerequisites

Install [uv](https://github.com/astral-sh/uv) if you haven't already:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/talmolab/labelroi.git
cd labelroi
```

2. Install the package using uv:
```bash
uv pip install -e .
```

This will install labelroi and all its dependencies in editable mode.

## Usage

### Command Line Interface

Launch the application with:
```bash
labelroi
```

Or open with a specific image/video:
```bash
labelroi path/to/image.png
labelroi path/to/video.mp4
```

### Drawing ROIs

1. Click **"Start Drawing ROI"** button
2. **Left-click** to add vertices to your polygon
3. Either:
   - Click on the first vertex (green dot) to close the polygon
   - **Right-click** to finish as a polyline (open shape)
4. Repeat to draw multiple ROIs (each gets a unique color)

### ROI Naming

- **Double-click** on any ROI to add or edit its name
- Names appear as white labels on the ROI
- Clear the name to remove the label

### Navigation Controls

**Mouse/Trackpad:**
- **Scroll wheel**: Zoom in/out
- **Pinch gesture**: Zoom in/out
- **Middle-click + drag**: Pan around
- **Ctrl + Left-click + drag**: Pan around
- **Alt + Double-click**: Reset zoom

**Keyboard:**
- **Arrow keys** or **WASD**: Pan
- **+/=**: Zoom in
- **-**: Zoom out
- **0** or **Home**: Reset zoom
- **F1** or **H**: Show help

### Saving and Loading ROIs

ROIs are automatically saved with the same name as your image/video file with a `.rois.yml` suffix:
- Image: `my_image.png` → ROIs: `my_image.rois.yml`
- Video: `my_video.mp4` → ROIs: `my_video.rois.yml`

Click **"Save ROIs to YAML"** to save your ROIs. They will automatically load when you open the same image/video again.

### YAML Format

The ROIs are saved in a human-readable YAML format:

```yaml
image_file: /path/to/image.png
roi_count: 2
rois:
  - id: 1
    name: "Region A"
    type: polygon
    coordinates:
      - [100.5, 200.3]
      - [150.2, 250.7]
      - [100.8, 300.1]
    color: "#1f77b4"
    properties:
      vertex_count: 3
      perimeter: 172.5
      area: 1250.3
  - id: 2
    name: "Line B"
    type: polyline
    coordinates:
      - [200.0, 100.0]
      - [300.0, 150.0]
    color: "#ff7f0e"
    properties:
      vertex_count: 2
      perimeter: 111.8
```

## Development

### Running from Source

```bash
# Clone and enter directory
git clone https://github.com/talmolab/labelroi.git
cd labelroi

# Install in development mode
uv pip install -e .

# Run the application
python src/labelroi/labelroi.py
```

### Dependencies

- **numpy**: Array operations
- **matplotlib**: Visualization and plotting
- **shapely**: Geometry operations
- **qtpy**: Qt compatibility layer
- **PyQt5**: Qt backend (can also use PyQt6, PySide2, or PySide6)
- **sleap-io**: Video file loading
- **PyYAML**: YAML file I/O

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems, please file an issue at https://github.com/talmolab/labelroi/issues