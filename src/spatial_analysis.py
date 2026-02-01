"""
Spatial Analysis Module
========================

Functions for detecting symmetry, geometry, edge orientation, and spatial patterns.
"""

import cv2
import numpy as np
from typing import Dict, Optional


def detect_symmetry(image_path: str) -> float:
    """
    Detect left-right symmetry in an artwork.
    
    Computes correlation between left and right halves of the image.
    Higher values (closer to 1.0) indicate greater symmetry.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    float
        Symmetry score between 0.0 and 1.0
        
    Examples:
    ---------
    >>> symmetry = detect_symmetry("artwork.jpg")
    >>> print(f"Symmetry score: {symmetry:.3f}")
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize for consistency
    img = cv2.resize(img, (200, 200))

    # Split and flip
    left_half = img[:, :img.shape[1]//2]
    right_half = img[:, img.shape[1]//2:]
    right_half_flipped = cv2.flip(right_half, 1)

    # Ensure same dimensions
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]

    # Calculate correlation
    correlation = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]

    return max(0.0, correlation)  # Clamp to 0-1


def edge_density(image_path: str) -> float:
    """
    Calculate edge density using Canny edge detection.
    
    Higher values indicate more complex, detailed artworks.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    float
        Edge density (ratio of edge pixels to total pixels)
        
    Examples:
    ---------
    >>> density = edge_density("artwork.jpg")
    >>> print(f"Edge density: {density:.4f}")
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def detect_edge_orientation(image_path: str) -> Dict:
    """
    Detect vertical vs horizontal orientation of visual mass.
    
    Uses Sobel gradients to determine if an artwork has predominantly
    vertical or horizontal structural elements.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    dict
        Dictionary with 'spatial_vertical_mass' and 'spatial_horizontal_mass'
        
    Examples:
    ---------
    >>> orientation = detect_edge_orientation("artwork.jpg")
    >>> print(f"Vertical: {orientation['spatial_vertical_mass']:.3f}")
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(img, 50, 150)

    # Sobel gradients
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate angles
    angles = np.arctan2(sobely, sobelx) * 180 / np.pi
    angles = np.abs(angles)

    # Count orientations
    vertical_mask = (angles > 80) & (angles < 100)
    horizontal_mask = (angles < 10) | (angles > 170)

    total_edges = np.sum(edges > 0)

    if total_edges == 0:
        return {'spatial_vertical_mass': 0.0, 'spatial_horizontal_mass': 0.0}

    vertical_pct = np.sum(vertical_mask) / total_edges
    horizontal_pct = np.sum(horizontal_mask) / total_edges

    return {
        'spatial_vertical_mass': round(vertical_pct, 3),
        'spatial_horizontal_mass': round(horizontal_pct, 3)
    }


def detect_shapes(image_path: str) -> Dict:
    """
    Detect geometric shapes (circles, rectangles, triangles) using contours.
    
    Combines Hough transforms and contour approximation to identify
    geometric patterns in artwork.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    dict
        Dictionary with geometry confidence scores:
        - geometry_circle_conf: 0-1 (circle detection confidence)
        - geometry_rectangle_conf: 0-1
        - geometry_square_conf: 0-1
        - geometry_triangle_conf: 0-1
        - geometry_line_density: number of lines per 10k pixels
        - geometry_undefined: 1 if no clear geometry, 0 otherwise
        - geometry_multiple_shapes: 1 if multiple shapes detected
        
    Examples:
    ---------
    >>> shapes = detect_shapes("geometric_artwork.jpg")
    >>> if shapes['geometry_circle_conf'] > 0.5:
    ...     print("Strong circular geometry detected!")
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    shape_metrics = {
        'geometry_circle_conf': 0.0,
        'geometry_rectangle_conf': 0.0,
        'geometry_square_conf': 0.0,
        'geometry_triangle_conf': 0.0,
        'geometry_line_density': 0.0,
        'geometry_undefined': 0,
        'geometry_multiple_shapes': 0
    }

    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=30, minRadius=10, maxRadius=200
    )
    if circles is not None:
        shape_metrics['geometry_circle_conf'] = min(len(circles[0]) / 5.0, 1.0)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        area = img.shape[0] * img.shape[1]
        shape_metrics['geometry_line_density'] = round(len(lines) / (area / 10000), 3)

    # Detect polygons
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = 0
    squares = 0
    triangles = 0

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        # Approximate polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            triangles += 1
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.9 <= aspect_ratio <= 1.1:
                squares += 1
            else:
                rectangles += 1

    shape_metrics['geometry_rectangle_conf'] = min(rectangles / 3.0, 1.0)
    shape_metrics['geometry_square_conf'] = min(squares / 3.0, 1.0)
    shape_metrics['geometry_triangle_conf'] = min(triangles / 3.0, 1.0)

    # Multiple shapes
    shape_count = sum([
        1 for v in [shape_metrics['geometry_circle_conf'],
                    shape_metrics['geometry_rectangle_conf'],
                    shape_metrics['geometry_square_conf'],
                    shape_metrics['geometry_triangle_conf']]
        if v > 0.3
    ])
    shape_metrics['geometry_multiple_shapes'] = 1 if shape_count > 2 else 0

    # Undefined geometry
    max_geometry = max([
        shape_metrics['geometry_circle_conf'],
        shape_metrics['geometry_rectangle_conf'],
        shape_metrics['geometry_square_conf'],
        shape_metrics['geometry_triangle_conf']
    ])
    shape_metrics['geometry_undefined'] = 1 if max_geometry < 0.2 else 0

    return shape_metrics
