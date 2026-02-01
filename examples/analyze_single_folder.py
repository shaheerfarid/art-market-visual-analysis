#!/usr/bin/env python3
"""
Quick Start Example: Analyze a Single Folder
============================================

This script demonstrates how to analyze a single folder of artworks.
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_image_files, get_folder_name
from src.color_analysis import extract_color_metrics
from src.spatial_analysis import detect_symmetry, detect_edge_orientation, detect_shapes
from src.semantic_analysis import initialize_clip_model, analyze_with_clip, get_dominant_medium
from src.sequence_analysis import calculate_pairwise_similarity, identify_curatorial_blocks


def analyze_folder(folder_path: str, output_path: str, use_clip: bool = True):
    """Analyze all images in a folder and export results."""
    
    print(f"\nArt Market Visual Analysis")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}")
    
    # Get image files
    images = get_image_files(folder_path)
    print(f"Found {len(images)} images")
    
    if len(images) == 0:
        print("No images found!")
        return
    
    # Initialize CLIP if needed
    if use_clip:
        print("\n Initializing CLIP model...")
        initialize_clip_model()
    
    # Analyze each image
    print(f"\nAnalyzing artworks...")
    results = []
    
    for idx, img_path in enumerate(tqdm(images, desc="Processing")):
        try:
            result = {
                'image_filename': img_path.name,
                'catalogue_position': idx + 1,
                'folder_name': get_folder_name(folder_path)
            }
            
            # Color analysis
            color_metrics = extract_color_metrics(str(img_path))
            result.update(color_metrics)
            
            # Spatial analysis
            result['symmetry_score'] = round(detect_symmetry(str(img_path)), 3)
            result.update(detect_edge_orientation(str(img_path)))
            result.update(detect_shapes(str(img_path)))
            
            # Semantic analysis (CLIP)
            if use_clip:
                clip_scores = analyze_with_clip(str(img_path))
                result.update(clip_scores)
                result['medium_primary'] = get_dominant_medium(clip_scores)
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError analyzing {img_path.name}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate sequence similarity
    print("\nCalculating sequence patterns...")
    feature_cols = [col for col in df.columns if col not in 
                   ['image_filename', 'catalogue_position', 'folder_name', 'medium_primary']]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]
    
    similarity_df = calculate_pairwise_similarity(df, feature_cols[:10])  # Use top 10 features
    
    # Identify blocks
    df_with_blocks = identify_curatorial_blocks(df, feature_cols[:10])
    
    # Export results
    print(f"\nExporting results to {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_with_blocks.to_excel(writer, sheet_name='Artwork Analysis', index=False)
        similarity_df.to_excel(writer, sheet_name='Sequence Similarity', index=False)
    
    # Display summary
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Total artworks analyzed: {len(df)}")
    print(f"\nColor averages:")
    print(f"  Primary colors: {df['color_primary_pct'].mean():.1f}%")
    print(f"  Secondary colors: {df['color_secondary_pct'].mean():.1f}%")
    print(f"  Tertiary colors: {df['color_tertiary_pct'].mean():.1f}%")
    
    if use_clip:
        print(f"\nContent distribution:")
        print(f"  Abstract: {df['abstract_general'].mean():.3f}")
        print(f"  Figurative: {df['figurative_general'].mean():.3f}")
        print(f"  Landscape: {df['landscape_general'].mean():.3f}")
    
    print(f"\nAverage similarity: {similarity_df['similarity_score'].mean():.3f}")
    print(f"Detected blocks: {df_with_blocks['block_id'].nunique()}")
    print(f"\n Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze artwork images in a folder"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to folder containing artwork images"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/analysis_results.xlsx",
        help="Output Excel file path (default: outputs/analysis_results.xlsx)"
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Skip CLIP semantic analysis (faster, but less information)"
    )
    
    args = parser.parse_args()
    
    analyze_folder(args.input, args.output, use_clip=not args.no_clip)
