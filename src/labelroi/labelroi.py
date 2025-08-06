#!/usr/bin/env python3
"""
ROI Labeling Application

A Qt application for drawing polygon ROIs on images and videos using qtpy compatibility layer.
Features:
- Display images from numpy arrays
- Display first frame from MP4 video files
- Draw polygon ROIs by clicking vertices
- Auto-close polygon when clicking on first vertex
- Right-click to keep as polyline
- Uses shapely for geometry handling
- Compatible with PyQt5/PySide2/PySide6 via qtpy
"""

import sys
import numpy as np
from typing import List, Optional, Tuple
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sleap_io as sio
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib import colors
import shapely.geometry as sg
from shapely import Polygon, LineString, Point

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QSpinBox,
    QMessageBox,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMenu,
    QLineEdit,
    QInputDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsProxyWidget,
    QPinchGesture,
    QGesture,
)
from qtpy.QtCore import Signal, Qt, QRectF, QEvent
from qtpy.QtGui import QMouseEvent, QWheelEvent, QColor

# Gesture support is always available in modern Qt
GESTURES_AVAILABLE = True


class ImageROIWidget(QGraphicsView):
    """Custom QGraphicsView for image display and ROI drawing with smooth navigation.

    This hybrid implementation uses QGraphicsView for smooth panning and zooming,
    while maintaining matplotlib for ROI drawing and display.
    """

    roi_created = Signal(object)  # Emits shapely geometry when ROI is created
    roi_name_changed = Signal(int)  # Emits ROI index when name is changed

    def __init__(self, parent=None, colormap="tab20"):
        super().__init__(parent)

        # Initialize graphics scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Configure view behavior
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Enable keyboard focus for navigation
        self.setFocusPolicy(Qt.StrongFocus)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Zoom tracking
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

        # Create matplotlib canvas for ROI drawing
        self.figure = Figure(figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")

        # Embed matplotlib canvas in graphics scene
        self.canvas_proxy = self.scene.addWidget(self.canvas)

        # Current drawing state
        self.current_points: List[Tuple[float, float]] = []
        self.drawing_roi = False
        self.first_point_marker = None
        self.current_line = None
        self.vertex_markers: List = []
        self.completed_rois: List[dict] = []

        # Text editing state
        self.text_annotations: List = []
        self.editing_roi_index = None
        self.last_click_time = 0
        self.double_click_threshold = 0.3

        # Color cycling setup
        self.colormap_name = colormap
        self.colormap = plt.get_cmap(colormap)
        self.color_index = 0

        # Visual settings
        self.current_roi_color = "red"
        self.roi_alpha = 0.3
        self.line_width = 2
        self.point_size = 8
        self.first_point_size = 12

        # Panning state for manual implementation
        self.panning = False
        self.pan_start_pos = None
        self.pan_start_center = None

        # Store original view for reset
        self.original_scene_rect = None

        # Connect matplotlib events for ROI drawing
        self.canvas.mpl_connect("button_press_event", self._on_mpl_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mpl_mouse_move)

        # Enable gestures
        self.grabGesture(Qt.PinchGesture)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

        # Image data
        self.image_data = None
        self.pixmap_item = None

    @property
    def roi_color(self):
        """Get the current ROI color for drawing."""
        return self.current_roi_color

    def _get_next_color(self):
        """Get the next color from the colormap for completed ROIs."""
        if hasattr(self.colormap, "colors"):
            num_colors = len(self.colormap.colors)
            color = self.colormap.colors[self.color_index % num_colors]
        else:
            color = self.colormap((self.color_index / 20.0) % 1.0)

        if isinstance(color, (list, tuple, np.ndarray)):
            color = colors.to_hex(color)

        self.color_index += 1
        return color

    def set_image(self, image_array: np.ndarray):
        """Set the image to display from a numpy array."""
        self.image_data = image_array.copy()
        self.ax.clear()
        self.text_annotations = []

        # Handle different image formats
        if len(image_array.shape) == 2:
            self.ax.imshow(image_array, cmap="gray")
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1:
                self.ax.imshow(image_array.squeeze(), cmap="gray")
            elif image_array.shape[2] in [3, 4]:
                if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                    if image_array.max() > 1.0:
                        image_array = image_array / 255.0
                self.ax.imshow(image_array)
            else:
                raise ValueError(
                    f"Unsupported number of channels: {image_array.shape[2]}"
                )
        else:
            raise ValueError(f"Unsupported image shape: {image_array.shape}")

        self.ax.set_title(
            "Click to draw ROI vertices. Right-click for polyline. Navigate with mouse/trackpad."
        )

        # Store original scene rect for reset
        self.canvas.draw()
        canvas_size = self.canvas.size()
        self.original_scene_rect = QRectF(
            0, 0, canvas_size.width(), canvas_size.height()
        )

        # Fit the view to the canvas
        self.fitInView(self.canvas_proxy, Qt.KeepAspectRatio)
        self.zoom_factor = 1.0

    def start_roi_drawing(self):
        """Start drawing a new ROI."""
        self.drawing_roi = True
        self.current_points = []
        if self.first_point_marker:
            self.first_point_marker.remove()
            self.first_point_marker = None
        if self.current_line:
            self.current_line.remove()
            self.current_line = None
        for marker in self.vertex_markers:
            marker.remove()
        self.vertex_markers = []
        self.canvas.draw()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for navigation."""
        # Store the click position for tracking drags
        self._down_pos = event.pos()

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier
        ):
            # Start manual panning since ScrollHandDrag doesn't work with embedded matplotlib
            self.panning = True
            self.pan_start_pos = event.pos()
            self.pan_start_center = self.mapToScene(self.rect().center())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier:
            # Regular left click - allow for ROI drawing and ensure focus
            self.panning = False
            self.setFocus()  # Ensure keyboard focus for navigation

        # Call super for other events
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events."""
        if self.panning and self.pan_start_pos is not None:
            # Calculate the movement delta
            delta = event.pos() - self.pan_start_pos

            # Convert delta to scene coordinates
            # We need to scale the delta by the current zoom level
            scale_factor = 1.0 / self.zoom_factor
            scene_delta_x = delta.x() * scale_factor
            scene_delta_y = delta.y() * scale_factor

            # Calculate new center position (invert delta since we're moving the view)
            new_center_x = self.pan_start_center.x() - scene_delta_x
            new_center_y = self.pan_start_center.y() - scene_delta_y

            # Apply the panning
            self.centerOn(new_center_x, new_center_y)
            event.accept()
            return

        # Let other events pass through
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        # Check if this was a drag operation
        has_moved = hasattr(self, "_down_pos") and event.pos() != self._down_pos

        if self.panning:
            # End panning
            self.panning = False
            self.pan_start_pos = None
            self.pan_start_center = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        # Call super to handle the event properly
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel - less sensitive and smoother."""
        # Only zoom when no mouse buttons are pressed
        if event.buttons() != Qt.NoButton:
            return

        # Get zoom direction with smaller, less sensitive factors
        angle = event.angleDelta().y()
        zoom_factor = 1.1 if angle > 0 else 0.9

        # Check zoom bounds more strictly
        new_zoom = self.zoom_factor * zoom_factor
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return

        # Apply zoom with proper bounds
        self.scale(zoom_factor, zoom_factor)
        self.zoom_factor = new_zoom

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double-click events for zoom reset."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.AltModifier:
            # Reset zoom on Alt+double-click
            self.zoom_factor = 1.0
            self.resetTransform()

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation."""
        # Pan with arrow keys
        pan_distance = 50  # pixels to pan
        scale_factor = 1.0 / self.zoom_factor
        scene_pan_distance = pan_distance * scale_factor

        current_center = self.mapToScene(self.rect().center())

        if event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
            # Pan left
            self.centerOn(current_center.x() - scene_pan_distance, current_center.y())
            event.accept()
        elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
            # Pan right
            self.centerOn(current_center.x() + scene_pan_distance, current_center.y())
            event.accept()
        elif event.key() == Qt.Key_Up or event.key() == Qt.Key_W:
            # Pan up
            self.centerOn(current_center.x(), current_center.y() - scene_pan_distance)
            event.accept()
        elif event.key() == Qt.Key_Down or event.key() == Qt.Key_S:
            # Pan down
            self.centerOn(current_center.x(), current_center.y() + scene_pan_distance)
            event.accept()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Zoom in
            if self.zoom_factor < self.max_zoom:
                zoom_factor = 1.1
                self.scale(zoom_factor, zoom_factor)
                self.zoom_factor *= zoom_factor
            event.accept()
        elif event.key() == Qt.Key_Minus:
            # Zoom out
            if self.zoom_factor > self.min_zoom:
                zoom_factor = 0.9
                self.scale(zoom_factor, zoom_factor)
                self.zoom_factor *= zoom_factor
            event.accept()
        elif event.key() == Qt.Key_0 or event.key() == Qt.Key_Home:
            # Reset zoom
            self.zoom_factor = 1.0
            self.resetTransform()
            event.accept()
        elif event.key() == Qt.Key_F1 or event.key() == Qt.Key_H:
            # Show help
            self.show_navigation_help()
            event.accept()
        else:
            super().keyPressEvent(event)

    def show_navigation_help(self):
        """Print navigation shortcuts to console."""
        help_text = """
Navigation Shortcuts:
===================
Mouse:
- Ctrl+Left Mouse Drag or Middle Mouse Drag: Pan
- Mouse Wheel: Zoom in/out
- Pinch Gesture: Zoom in/out
- Alt+Double Click: Reset zoom

Keyboard:
- Arrow Keys or WASD: Pan
- +/= : Zoom in
- - : Zoom out  
- 0 or Home: Reset zoom
- F1 or H: Show this help

ROI Drawing:
- Left Click: Add vertex to polygon
- Right Click: Complete polyline (no auto-close)
- Click on first vertex: Complete polygon (auto-close)
- Double Click on ROI: Edit name
"""
        print(help_text)

    def event(self, event):
        """Handle gesture events."""
        if event.type() == QEvent.Gesture:
            return self._handle_gesture_event(event)
        return super().event(event)

    def _handle_gesture_event(self, event):
        """Handle gesture events, particularly pinch gestures."""
        pinch = event.gesture(Qt.PinchGesture)
        if pinch:
            return self._handle_pinch_gesture(pinch)
        return False

    def _handle_pinch_gesture(self, gesture: QPinchGesture):
        """Handle pinch gesture for zooming - simplified and more robust."""
        if gesture.state() == Qt.GestureUpdated:
            # Get the scale factor from the gesture
            scale_factor = gesture.scaleFactor()

            # Apply bounds checking similar to wheel zoom
            new_zoom = self.zoom_factor * scale_factor
            if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
                return True

            # Apply the scaling transformation
            self.scale(scale_factor, scale_factor)
            self.zoom_factor = new_zoom

            return True
        return False

    def _on_mpl_mouse_press(self, event):
        """Handle matplotlib mouse press events for ROI drawing."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Check for double-click on existing ROI for naming
        import time

        current_time = time.time()
        if (
            event.button == 1
            and (current_time - self.last_click_time) < self.double_click_threshold
        ):
            roi_index = self._find_roi_at_point(x, y)
            if roi_index is not None:
                self._start_editing_roi_name(roi_index, x, y)
                return

        self.last_click_time = current_time

        # ROI drawing only with left click when drawing mode is active
        if not self.drawing_roi:
            return

        # Check if we clicked near the first point to close the polygon
        if len(self.current_points) >= 3 and event.button == 1:
            first_point = Point(self.current_points[0])
            click_point = Point(x, y)
            if first_point.distance(click_point) < 10:
                self._complete_roi(is_polygon=True)
                return

        # Add new point
        self.current_points.append((x, y))

        # Handle right click - complete as polyline
        if event.button == 3:
            if len(self.current_points) >= 2:
                self._complete_roi(is_polygon=False)
            return

        self._update_drawing()

    def _on_mpl_mouse_move(self, event):
        """Handle matplotlib mouse move events for ROI preview."""
        if not self.drawing_roi or not self.current_points or event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Update preview line from last point to current mouse position
        if self.current_line:
            self.current_line.remove()

        last_point = self.current_points[-1]
        self.current_line = Line2D(
            [last_point[0], x],
            [last_point[1], y],
            color=self.roi_color,
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
        self.ax.add_line(self.current_line)
        self.canvas.draw_idle()

    def _find_roi_at_point(self, x, y):
        """Find the ROI index that contains the given point."""
        click_point = Point(x, y)

        # Check ROIs in reverse order (most recently added first)
        for i in reversed(range(len(self.completed_rois))):
            roi_data = self.completed_rois[i]
            geometry = roi_data["geometry"]

            # For polygons, check if point is inside
            if roi_data["is_polygon"] and geometry.contains(click_point):
                return i
            # For polylines, check if point is close to the line
            elif not roi_data["is_polygon"]:
                distance = geometry.distance(click_point)
                if distance < 5:  # 5 pixel tolerance for polylines
                    return i

        return None

    def _start_editing_roi_name(self, roi_index, x, y):
        """Start editing the name of the specified ROI."""
        if self.editing_roi_index is not None:
            return  # Already editing another ROI

        self.editing_roi_index = roi_index
        roi_data = self.completed_rois[roi_index]
        current_name = roi_data.get("name", "")

        # Get centroid of the ROI for text placement
        geometry = roi_data["geometry"]
        if roi_data["is_polygon"]:
            centroid = geometry.centroid
        else:
            # For polylines, use the midpoint
            coords = list(geometry.coords)
            mid_idx = len(coords) // 2
            centroid = Point(coords[mid_idx])

        # Use Qt input dialog for text editing
        self._show_name_input_dialog(roi_index, current_name, centroid.x, centroid.y)

    def _show_name_input_dialog(self, roi_index, current_name, x, y):
        """Show input dialog for editing ROI name."""
        from qtpy.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            None, f"Edit ROI #{roi_index + 1} Name", "ROI Name:", text=current_name
        )

        if ok:
            self._update_roi_name(roi_index, text)

        self.editing_roi_index = None

    def _update_roi_name(self, roi_index, name):
        """Update the name of the specified ROI."""
        if 0 <= roi_index < len(self.completed_rois):
            self.completed_rois[roi_index]["name"] = name
            self._update_roi_text_display(roi_index)
            self.canvas.draw()
            # Emit signal to update the table
            self.roi_name_changed.emit(roi_index)

    def _update_roi_text_display(self, roi_index):
        """Update the text display for a specific ROI."""
        if roi_index >= len(self.text_annotations):
            # Extend text annotations list if needed
            while len(self.text_annotations) <= roi_index:
                self.text_annotations.append(None)

        # Remove existing text annotation if it exists
        if self.text_annotations[roi_index] is not None:
            self.text_annotations[roi_index].remove()

        roi_data = self.completed_rois[roi_index]
        name = roi_data.get("name", "")

        if name:  # Only add text if there's a name
            geometry = roi_data["geometry"]

            # Get centroid for text placement
            if roi_data["is_polygon"]:
                centroid = geometry.centroid
            else:
                # For polylines, use the midpoint
                coords = list(geometry.coords)
                mid_idx = len(coords) // 2
                centroid = Point(coords[mid_idx])

            # Add text annotation
            text_obj = self.ax.text(
                centroid.x,
                centroid.y,
                name,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                ),
                picker=True,
            )
            self.text_annotations[roi_index] = text_obj
        else:
            self.text_annotations[roi_index] = None

    def reset_view(self):
        """Reset the view to fit the entire canvas."""
        if self.original_scene_rect:
            self.fitInView(self.canvas_proxy, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0

    def _update_drawing(self):
        """Update the visual representation of the current ROI being drawn."""
        if not self.current_points:
            return

        # Remove previous preview line
        if self.current_line:
            self.current_line.remove()
            self.current_line = None

        # Clear existing vertex markers
        for marker in self.vertex_markers:
            marker.remove()
        self.vertex_markers = []

        # Add vertex markers for each point
        for i, point in enumerate(self.current_points):
            if i == 0 and len(self.current_points) >= 3:
                continue  # Skip first point if it will be highlighted separately
            marker = self.ax.plot(point[0], point[1], "ro", markersize=self.point_size)[
                0
            ]
            self.vertex_markers.append(marker)

        # Draw lines between points
        if len(self.current_points) >= 2:
            points = np.array(self.current_points)
            line = Line2D(
                points[:, 0],
                points[:, 1],
                color=self.roi_color,
                linewidth=self.line_width,
            )
            self.ax.add_line(line)

        # Highlight first point if we have enough points for potential closure
        if len(self.current_points) >= 3:
            if self.first_point_marker:
                self.first_point_marker.remove()
            first_point = self.current_points[0]
            self.first_point_marker = self.ax.plot(
                first_point[0], first_point[1], "go", markersize=self.first_point_size
            )[0]

        self.canvas.draw()

    def _complete_roi(self, is_polygon: bool):
        """Complete the current ROI and create shapely geometry."""
        if len(self.current_points) < (3 if is_polygon else 2):
            return

        # Create shapely geometry
        if is_polygon:
            points = self.current_points.copy()
            if points[0] != points[-1]:
                points.append(points[0])
            geometry = Polygon(points)
        else:
            geometry = LineString(self.current_points)

        # Create visual representation
        points_array = np.array(self.current_points)
        roi_display_color = self._get_next_color()

        if is_polygon:
            closed_points = np.append(points_array, [points_array[0]], axis=0)
            poly_patch = MplPolygon(
                points_array,
                facecolor=roi_display_color,
                alpha=self.roi_alpha,
                edgecolor="none",
                linewidth=0,
            )
            self.ax.add_patch(poly_patch)
            edge_line = Line2D(
                closed_points[:, 0],
                closed_points[:, 1],
                color=roi_display_color,
                linewidth=self.line_width,
            )
            self.ax.add_line(edge_line)
        else:
            line = Line2D(
                points_array[:, 0],
                points_array[:, 1],
                color=roi_display_color,
                linewidth=self.line_width,
            )
            self.ax.add_line(line)

        # Store ROI data
        roi_data = {
            "geometry": geometry,
            "points": self.current_points.copy(),
            "is_polygon": is_polygon,
            "color": roi_display_color,
            "name": "",
        }
        self.completed_rois.append(roi_data)
        self.text_annotations.append(None)

        # Clean up drawing state
        self.drawing_roi = False
        self.current_points = []
        if self.first_point_marker:
            self.first_point_marker.remove()
            self.first_point_marker = None
        if self.current_line:
            self.current_line.remove()
            self.current_line = None
        for marker in self.vertex_markers:
            marker.remove()
        self.vertex_markers = []

        self.canvas.draw()
        self.roi_created.emit(geometry)

    def clear_rois(self):
        """Clear all ROIs."""
        self.completed_rois = []
        for text_obj in self.text_annotations:
            if text_obj is not None:
                text_obj.remove()
        self.text_annotations = []

        if self.image_data is not None:
            self.set_image(self.image_data)
        else:
            self.ax.clear()
            self.canvas.draw()

    def delete_roi(self, roi_index):
        """Delete a specific ROI by index."""
        if 0 <= roi_index < len(self.completed_rois):
            if (
                roi_index < len(self.text_annotations)
                and self.text_annotations[roi_index] is not None
            ):
                self.text_annotations[roi_index].remove()
                self.text_annotations.pop(roi_index)

            self.completed_rois.pop(roi_index)
            self.color_index = len(self.completed_rois)

            if self.image_data is not None:
                self.set_image(self.image_data)
                for i, roi_data in enumerate(self.completed_rois):
                    self._redraw_roi(roi_data, i)
                    self._update_roi_text_display(i)
            self.canvas.draw()

    def get_rois(self) -> List[dict]:
        """Get all completed ROIs."""
        return self.completed_rois.copy()

    def _redraw_roi(self, roi_data, color_index):
        """Redraw a specific ROI with the given color index."""
        geometry = roi_data["geometry"]
        is_polygon = roi_data["is_polygon"]
        points_array = np.array(roi_data["points"])

        if hasattr(self.colormap, "colors"):
            num_colors = len(self.colormap.colors)
            color = self.colormap.colors[color_index % num_colors]
        else:
            color = self.colormap((color_index / 20.0) % 1.0)

        if isinstance(color, (list, tuple, np.ndarray)):
            color = colors.to_hex(color)

        roi_data["color"] = color

        if is_polygon:
            closed_points = np.append(points_array, [points_array[0]], axis=0)
            poly_patch = MplPolygon(
                points_array,
                facecolor=color,
                alpha=self.roi_alpha,
                edgecolor="none",
                linewidth=0,
            )
            self.ax.add_patch(poly_patch)
            edge_line = Line2D(
                closed_points[:, 0],
                closed_points[:, 1],
                color=color,
                linewidth=self.line_width,
            )
            self.ax.add_line(edge_line)
        else:
            line = Line2D(
                points_array[:, 0],
                points_array[:, 1],
                color=color,
                linewidth=self.line_width,
            )
            self.ax.add_line(line)

    def _redraw_all_rois(self):
        """Redraw all completed ROIs on the current image."""
        if self.image_data is not None:
            self.set_image(self.image_data)
            for i, roi_data in enumerate(self.completed_rois):
                self._redraw_roi(roi_data, i)
                self._update_roi_text_display(i)
            self.canvas.draw()


class ROILabelingApp(QMainWindow):
    """Main application window."""

    def __init__(self, image_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("ROI Labeling Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Track current image path for saving
        self.current_image_path: Optional[str] = image_path

        self.init_ui()

        # Load image if path provided, otherwise create sample
        if image_path:
            self.load_image_from_path(image_path)
        else:
            self.create_sample_image()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel for image
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel)
        image_layout = QVBoxLayout(image_frame)

        # Image widget
        self.image_widget = ImageROIWidget()
        self.image_widget.roi_created.connect(self.on_roi_created)
        self.image_widget.roi_name_changed.connect(self.on_roi_name_changed)
        image_layout.addWidget(self.image_widget)

        # Right panel for controls
        control_panel = QWidget()
        control_panel.setMinimumWidth(350)  # Increased width for table
        control_panel.setMaximumWidth(400)
        control_layout = QVBoxLayout(control_panel)

        # Load image/video button
        load_btn = QPushButton("Load Image/Video")
        load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(load_btn)

        # Sample image button
        sample_btn = QPushButton("Create Sample Image")
        sample_btn.clicked.connect(self.create_sample_image)
        control_layout.addWidget(sample_btn)

        # ROI controls
        control_layout.addWidget(QLabel("ROI Controls:"))

        draw_btn = QPushButton("Start Drawing ROI")
        draw_btn.clicked.connect(self.start_roi_drawing)
        control_layout.addWidget(draw_btn)

        clear_btn = QPushButton("Clear All ROIs")
        clear_btn.clicked.connect(self.clear_rois)
        control_layout.addWidget(clear_btn)

        delete_btn = QPushButton("Delete Selected ROI")
        delete_btn.clicked.connect(self.delete_selected_roi)
        control_layout.addWidget(delete_btn)

        # Save ROIs button
        save_btn = QPushButton("Save ROIs to YAML")
        save_btn.clicked.connect(self.save_rois)
        control_layout.addWidget(save_btn)

        # Navigation controls
        control_layout.addWidget(QLabel("View Controls:"))

        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(reset_view_btn)

        # ROI info
        self.roi_info_label = QLabel("ROIs: 0")
        control_layout.addWidget(self.roi_info_label)

        # ROI table
        control_layout.addWidget(QLabel("ROI Properties:"))
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(7)
        self.roi_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Type", "Vertices", "Perimeter", "Area", "Color"]
        )

        # Configure table appearance
        header = self.roi_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.roi_table.setAlternatingRowColors(True)
        self.roi_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.roi_table.setMaximumHeight(200)
        self.roi_table.setMinimumHeight(100)

        # Enable context menu
        self.roi_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_table.customContextMenuRequested.connect(self.show_roi_context_menu)

        control_layout.addWidget(self.roi_table)

        # Instructions
        instructions = QLabel(
            "Instructions:\n\n"
            "ROI Drawing:\n"
            "1. Click 'Start Drawing ROI'\n"
            "2. Left-click to add vertices\n"
            "3. Click on first vertex to close polygon\n"
            "4. Right-click to finish as polyline\n"
            "5. Repeat for multiple ROIs\n\n"
            "ROI Naming:\n"
            "• Double-click on any ROI to edit its name\n"
            "• Type the name and press Enter to save\n"
            "• Names appear in white boxes on the ROI\n"
            "• Empty names hide the text display\n\n"
            "ROI Management:\n"
            "• View ROI properties in the table below\n"
            "• Right-click table rows for options\n"
            "• Select row and click 'Delete Selected ROI'\n"
            "• 'Clear All ROIs' removes everything\n\n"
            "Navigation:\n"
            "• Scroll wheel or pinch to zoom\n"
            "• Two-finger pan to move around\n"
            "• Middle-click + drag to pan\n"
            "• Ctrl+Left-click + drag to pan\n"
            "• 'Reset View' to fit image"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 10px; }"
        )
        control_layout.addWidget(instructions)

        control_layout.addStretch()

        # Add panels to main layout
        main_layout.addWidget(image_frame, 3)
        main_layout.addWidget(control_panel, 1)

    def load_image(self):
        """Load an image or video from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image or Video",
            "",
            "Image/Video Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.mp4);;Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Video Files (*.mp4)",
        )

        if file_path:
            try:
                # Store the current image path
                self.current_image_path = file_path
                # Load image or video
                image_array = self._load_media_file(file_path)
                self.image_widget.set_image(image_array)

                # Try to load ROIs from corresponding YAML file
                self.load_rois_if_available()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load media: {str(e)}")

    def _load_media_file(self, file_path: str) -> np.ndarray:
        """Load an image or video file and return the image array.

        For video files, returns the first frame.
        For image files, returns the image array.
        """
        file_path_lower = file_path.lower()

        if file_path_lower.endswith(".mp4"):
            # Load video and get first frame
            try:
                video = sio.load_video(file_path, grayscale=False)
                image_array = video[0]  # Get first frame
                return image_array
            except Exception as e:
                raise Exception(f"Failed to load video file: {str(e)}")
        else:
            # Load as image using matplotlib
            import matplotlib.image as mpimg

            return mpimg.imread(file_path)

    def load_image_from_path(self, file_path: str):
        """Load an image or video from a given file path."""
        try:
            # Store the current image path
            self.current_image_path = file_path
            # Load image or video
            image_array = self._load_media_file(file_path)
            self.image_widget.set_image(image_array)

            # Try to load ROIs from corresponding YAML file
            self.load_rois_if_available()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load media: {str(e)}")
            # Fall back to sample image if loading fails
            self.create_sample_image()

    def create_sample_image(self):
        """Create a sample image for testing."""
        # Clear current image path since this is a sample
        self.current_image_path = None

        # Create a colorful test image
        height, width = 400, 600

        # Create gradient background
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)

        # RGB channels
        R = np.sin(2 * np.pi * X) * 0.5 + 0.5
        G = np.cos(2 * np.pi * Y) * 0.5 + 0.5
        B = np.sin(2 * np.pi * (X + Y)) * 0.5 + 0.5

        # Combine channels
        image_array = np.stack([R, G, B], axis=2)

        # Add some geometric shapes for reference
        center_x, center_y = width // 2, height // 2
        y_indices, x_indices = np.ogrid[:height, :width]

        # Add a circle
        circle_mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 < 50**2
        image_array[circle_mask] = [1, 1, 1]  # White circle

        # Add some rectangles
        image_array[100:150, 100:200] = [1, 0, 0]  # Red rectangle
        image_array[250:300, 400:500] = [0, 1, 0]  # Green rectangle

        self.image_widget.set_image(image_array)

    def start_roi_drawing(self):
        """Start drawing a new ROI."""
        self.image_widget.start_roi_drawing()

    def clear_rois(self):
        """Clear all ROIs."""
        self.image_widget.clear_rois()
        self.update_roi_info()

    def reset_view(self):
        """Reset the view to fit the entire image."""
        self.image_widget.reset_view()

    def on_roi_created(self, geometry):
        """Handle ROI creation."""
        self.update_roi_info()

        # Print geometry info
        if isinstance(geometry, Polygon):
            print(f"Created polygon with area: {geometry.area:.2f}")
            print(f"Polygon coordinates: {list(geometry.exterior.coords)}")
        elif isinstance(geometry, LineString):
            print(f"Created polyline with length: {geometry.length:.2f}")
            print(f"Polyline coordinates: {list(geometry.coords)}")

    def on_roi_name_changed(self, roi_index):
        """Handle ROI name changes."""
        self.update_roi_info()  # This will update the table with the new name

    def update_roi_info(self):
        """Update ROI information display."""
        rois = self.image_widget.get_rois()
        polygons = sum(1 for roi in rois if roi["is_polygon"])
        polylines = sum(1 for roi in rois if not roi["is_polygon"])

        self.roi_info_label.setText(
            f"ROIs: {len(rois)}\n" f"Polygons: {polygons}\n" f"Polylines: {polylines}"
        )

        # Update the table
        self.update_roi_table()

    def update_roi_table(self):
        """Update the ROI properties table."""
        rois = self.image_widget.get_rois()
        self.roi_table.setRowCount(len(rois))

        for i, roi in enumerate(rois):
            geometry = roi["geometry"]
            is_polygon = roi["is_polygon"]

            # ROI ID
            id_item = QTableWidgetItem(str(i + 1))
            id_item.setToolTip(f"ROI #{i + 1}")
            self.roi_table.setItem(i, 0, id_item)

            # Name
            name = roi.get("name", "")
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(f"ROI name: {name if name else 'Unnamed'}")
            self.roi_table.setItem(i, 1, name_item)

            # Type
            roi_type = "Polygon" if is_polygon else "Polyline"
            type_item = QTableWidgetItem(roi_type)
            type_item.setToolTip(f"Geometry type: {roi_type}")
            self.roi_table.setItem(i, 2, type_item)

            # Vertex count
            vertex_count = len(roi["points"])
            vertex_item = QTableWidgetItem(str(vertex_count))
            vertex_item.setToolTip(f"Number of vertices: {vertex_count}")
            self.roi_table.setItem(i, 3, vertex_item)

            # Perimeter (or length for polylines)
            if is_polygon:
                perimeter = geometry.length
                perimeter_str = f"{perimeter:.1f}"
            else:
                length = geometry.length
                perimeter_str = f"{length:.1f}"
            perimeter_item = QTableWidgetItem(perimeter_str)
            perimeter_item.setToolTip(f"Perimeter/Length: {perimeter_str} pixels")
            self.roi_table.setItem(i, 4, perimeter_item)

            # Area (only for polygons)
            if is_polygon:
                area = geometry.area
                area_str = f"{area:.1f}"
                area_item = QTableWidgetItem(area_str)
                area_item.setToolTip(f"Area: {area_str} square pixels")
            else:
                area_str = "N/A"
                area_item = QTableWidgetItem(area_str)
                area_item.setToolTip("Area not applicable for polylines")
            self.roi_table.setItem(i, 5, area_item)

            # Color (from stored ROI data)
            color_hex = roi.get("color", "#000000")
            color_item = QTableWidgetItem(color_hex)
            color_item.setToolTip(f"Display color: {color_hex}")

            # Set background color to match the ROI color
            try:
                qcolor = QColor(color_hex)
                color_item.setBackground(qcolor)
                # Use white text if the color is dark
                if qcolor.lightness() < 128:
                    color_item.setForeground(QColor("white"))
            except:
                pass  # If color parsing fails, just show the hex string

            self.roi_table.setItem(i, 6, color_item)

    def show_roi_context_menu(self, position):
        """Show context menu for ROI table."""
        if self.roi_table.itemAt(position) is not None:
            menu = QMenu()

            delete_action = menu.addAction("Delete ROI")
            delete_action.triggered.connect(lambda: self.delete_selected_roi())

            menu.exec_(self.roi_table.mapToGlobal(position))

    def delete_selected_roi(self):
        """Delete the currently selected ROI."""
        current_row = self.roi_table.currentRow()
        if current_row >= 0:
            reply = QMessageBox.question(
                self,
                "Delete ROI",
                f"Are you sure you want to delete ROI #{current_row + 1}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.image_widget.delete_roi(current_row)
                self.update_roi_info()

    def load_rois_if_available(self):
        """Load ROIs from YAML file if it exists next to the current image."""
        if not self.current_image_path:
            return

        # Generate expected ROI file path
        image_path = Path(self.current_image_path)
        roi_path = image_path.with_suffix(".rois.yml")

        if roi_path.exists():
            try:
                self.load_rois_from_yaml(roi_path)
                print(f"Loaded ROIs from: {roi_path}")
            except Exception as e:
                print(f"Failed to load ROIs from {roi_path}: {e}")
                # Don't show error dialog - just log it silently

    def load_rois_from_yaml(self, yaml_path):
        """Load ROIs from a YAML file and add them to the image widget."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if "rois" not in data:
            raise ValueError("No 'rois' key found in YAML file")

        # Clear existing ROIs
        self.image_widget.clear_rois()

        # Load each ROI
        for roi_data in data["rois"]:
            coords = roi_data["coordinates"]
            roi_type = roi_data["type"]
            name = roi_data.get("name", "")
            color = roi_data.get("color", "#ff0000")

            # Convert coordinates to proper format
            points = [(float(x), float(y)) for x, y in coords]

            # Create geometry
            if roi_type == "polygon":
                geometry = Polygon(points)
                is_polygon = True
            else:  # polyline
                geometry = LineString(points)
                is_polygon = False

            # Add ROI to the image widget
            roi_dict = {
                "geometry": geometry,
                "points": points,
                "is_polygon": is_polygon,
                "name": name,
                "color": color,
            }

            # Add to completed ROIs list
            self.image_widget.completed_rois.append(roi_dict)

        # Redraw all ROIs and update UI
        self.image_widget._redraw_all_rois()
        self.update_roi_info()

    def save_rois(self):
        """Save ROIs to a YAML file."""
        rois = self.image_widget.get_rois()

        if not rois:
            QMessageBox.information(self, "No ROIs", "No ROIs to save.")
            return

        # Determine output filename
        if self.current_image_path:
            # Use same path as image but with .rois.yml extension
            image_path = Path(self.current_image_path)
            output_path = image_path.with_suffix(".rois.yml")
        else:
            # No image loaded, ask user for filename
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save ROIs", "rois.yml", "YAML Files (*.yml *.yaml)"
            )
            if not output_path:
                return  # User cancelled
            output_path = Path(output_path)

        try:
            # Convert ROI data to serializable format
            roi_data = []
            for i, roi in enumerate(rois):
                geometry = roi["geometry"]

                # Convert shapely geometry to coordinates
                if roi["is_polygon"]:
                    # For polygons, get exterior coordinates
                    coords = [
                        list(coord) for coord in geometry.exterior.coords[:-1]
                    ]  # Remove duplicate last point and convert to lists
                    roi_type = "polygon"
                else:
                    # For polylines, get all coordinates
                    coords = [
                        list(coord) for coord in geometry.coords
                    ]  # Convert to lists
                    roi_type = "polyline"

                roi_dict = {
                    "id": i + 1,
                    "name": roi.get("name", ""),
                    "type": roi_type,
                    "coordinates": coords,
                    "color": roi.get("color", "#000000"),
                    "properties": {
                        "vertex_count": len(coords),
                        "perimeter": float(geometry.length),
                    },
                }

                # Add area for polygons
                if roi["is_polygon"]:
                    roi_dict["properties"]["area"] = float(geometry.area)

                roi_data.append(roi_dict)

            # Create final YAML structure
            yaml_data = {
                "image_file": (
                    str(self.current_image_path) if self.current_image_path else None
                ),
                "roi_count": len(roi_data),
                "rois": roi_data,
                "metadata": {
                    "created_with": "ROI Labeling Tool",
                    "format_version": "1.0",
                },
            }

            # Save to YAML file
            with open(output_path, "w") as f:
                yaml.dump(
                    yaml_data, f, default_flow_style=False, sort_keys=False, indent=2
                )

            QMessageBox.information(
                self,
                "ROIs Saved",
                f"Successfully saved {len(roi_data)} ROI(s) to:\n{output_path}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save ROIs: {str(e)}")


def main():
    """Run the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ROI Labeling Tool")
    parser.add_argument(
        "image",
        nargs="?",
        type=str,
        help="Path to image file to load on startup (optional)",
    )

    # Parse arguments, but remove them from sys.argv so QApplication doesn't see them
    args = parser.parse_args()

    # Remove our custom arguments from sys.argv for QApplication
    # Keep only the script name and any Qt-specific arguments
    qt_argv = [sys.argv[0]]
    sys.argv = qt_argv

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = ROILabelingApp(image_path=args.image)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
