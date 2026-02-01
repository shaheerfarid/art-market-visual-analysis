"""
Utility Functions
=================

Helper functions for image loading, preprocessing, and file management.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import os


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Load an image from file path and convert to RGB.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    target_size : tuple, optional
        Target (width, height) for resizing. If None, original size is kept.
    
    Returns:
    --------
    np.ndarray or None
        RGB image array, or None if loading failed
        
    Examples:
    ---------
    >>> img = load_image("artwork.jpg")
    >>> img_resized = load_image("artwork.jpg", target_size=(400, 400))
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if requested
        if target_size is not None:
            img = cv2.resize(img, target_size)
        
        return img
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_image_files(folder_path: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Get all image files from a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing images
    extensions : list, optional
        List of valid file extensions. Default: ['.jpg', '.jpeg', '.png', '.bmp']
    
    Returns:
    --------
    list of Path
        Sorted list of image file paths
        
    Examples:
    ---------
    >>> images = get_image_files("data/auction_2023/")
    >>> print(f"Found {len(images)} images")
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def resize_for_efficiency(img: np.ndarray, max_dimension: int = 400) -> np.ndarray:
    """
    Resize image for faster processing while maintaining aspect ratio.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image array
    max_dimension : int
        Maximum dimension (width or height)
    
    Returns:
    --------
    np.ndarray
        Resized image
        
    Examples:
    ---------
    >>> img = load_image("large_artwork.jpg")
    >>> img_small = resize_for_efficiency(img, max_dimension=400)
    """
    h, w = img.shape[:2]
    
    if max(h, w) <= max_dimension:
        return img
    
    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_output_directory(output_path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Parameters:
    -----------
    output_path : str
        Path to output file or directory
        
    Examples:
    ---------
    >>> ensure_output_directory("outputs/reports/results.xlsx")
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def get_folder_name(folder_path: str) -> str:
    """
    Extract folder name from path.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder
    
    Returns:
    --------
    str
        Folder name
        
    Examples:
    ---------
    >>> name = get_folder_name("/data/Auction_2023_Fall/")
    >>> print(name)  # "Auction_2023_Fall"
    """
    return Path(folder_path).name
