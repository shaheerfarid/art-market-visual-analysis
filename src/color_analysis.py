"""
Color Analysis Module
=====================

Functions for extracting color metrics, HSV analysis, and color theory categorization.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple
from collections import Counter


def rgb_to_hsv_category(rgb: np.ndarray) -> Tuple[str, str]:
    """
    Convert RGB color to named color category and type (primary/secondary/tertiary).
    
    Parameters:
    -----------
    rgb : np.ndarray
        RGB color values [R, G, B]
    
    Returns:
    --------
    tuple of (str, str)
        (named_color, color_type) e.g., ('red', 'primary')
        
    Examples:
    ---------
    >>> color, type_ = rgb_to_hsv_category(np.array([255, 0, 0]))
    >>> print(color, type_)  # 'red', 'primary'
    """
    # Convert to HSV
    rgb_norm = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2HSV)[0][0]
    hue = hsv[0] * 2  # OpenCV uses 0-180, convert to 0-360
    sat = hsv[1]
    val = hsv[2]

    # Check for achromatic colors first
    if sat < 30:  # Low saturation
        if val > 200:
            return 'white', None
        elif val < 50:
            return 'black', None
        else:
            return 'gray', None

    # Determine named color based on hue
    if 350 <= hue or hue <= 10:
        named_color = 'red'
    elif 10 < hue <= 40:
        named_color = 'orange'
    elif 40 < hue <= 75:
        named_color = 'yellow'
    elif 75 < hue <= 150:
        named_color = 'green'
    elif 150 < hue <= 250:
        named_color = 'blue'
    elif 250 < hue <= 330:
        named_color = 'purple'
    else:
        named_color = 'magenta'

    # Determine color type (primary/secondary/tertiary)
    if named_color in ['red', 'yellow', 'blue']:
        color_type = 'primary'
    elif named_color in ['orange', 'green', 'purple']:
        color_type = 'secondary'
    else:
        color_type = 'tertiary'

    return named_color, color_type


def extract_color_metrics(image_path: str, n_clusters: int = 8) -> Dict:
    """
    Extract comprehensive color metrics from an image using K-Means clustering.
    
    This function analyzes the color distribution of an artwork by:
    1. Clustering pixels into dominant colors (K-Means)
    2. Categorizing each cluster by hue (red, blue, yellow, etc.)
    3. Computing percentages for color theory categories (primary, secondary, tertiary)
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    n_clusters : int, default=8
        Number of color clusters for K-Means
    
    Returns:
    --------
    dict
        Dictionary containing color percentages:
        - color_blue_pct, color_yellow_pct, color_red_pct, etc.
        - color_primary_pct, color_secondary_pct, color_tertiary_pct
        
    Examples:
    ---------
    >>> metrics = extract_color_metrics("artwork.jpg")
    >>> print(f"Blue: {metrics['color_blue_pct']:.1f}%")
    >>> print(f"Primary colors: {metrics['color_primary_pct']:.1f}%")
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for faster processing
    h, w = img.shape[:2]
    max_dim = 400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Reshape for clustering
    pixels = img.reshape(-1, 3)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get cluster centers and their frequencies
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = Counter(labels)
    total_pixels = len(labels)

    # Initialize color metrics
    color_metrics = {
        'color_blue_pct': 0.0,
        'color_yellow_pct': 0.0,
        'color_white_pct': 0.0,
        'color_black_pct': 0.0,
        'color_red_pct': 0.0,
        'color_green_pct': 0.0,
        'color_orange_pct': 0.0,
        'color_purple_pct': 0.0,
        'color_gray_pct': 0.0,
        'color_primary_pct': 0.0,
        'color_secondary_pct': 0.0,
        'color_tertiary_pct': 0.0
    }

    # Categorize each cluster
    for cluster_id, center in enumerate(centers):
        freq = counts[cluster_id] / total_pixels
        named_color, color_type = rgb_to_hsv_category(center)

        # Add to named color
        color_key = f'color_{named_color}_pct'
        if color_key in color_metrics:
            color_metrics[color_key] += freq

        # Add to color type
        if color_type:
            type_key = f'color_{color_type}_pct'
            color_metrics[type_key] += freq

    # Convert to percentages
    for key in color_metrics:
        color_metrics[key] = round(color_metrics[key] * 100, 2)

    return color_metrics


def primary_color_percentage(color_dict: Dict) -> float:
    """
    Calculate total percentage of primary colors (red, blue, yellow).
    
    Parameters:
    -----------
    color_dict : dict
        Dictionary with color percentages
    
    Returns:
    --------
    float
        Total percentage of primary colors
        
    Examples:
    ---------
    >>> colors = extract_color_metrics("artwork.jpg")
    >>> primary = primary_color_percentage(colors)
    """
    return color_dict.get("color_red_pct", 0) + \
           color_dict.get("color_blue_pct", 0) + \
           color_dict.get("color_yellow_pct", 0)
