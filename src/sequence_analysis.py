"""
Sequence Analysis Module
=========================

Functions for analyzing temporal patterns, detecting visual transitions,
and identifying curatorial blocks in artwork sequences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine


def calculate_pairwise_similarity(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Calculate cosine similarity between consecutive artworks in a sequence.
    
    This function measures how visually similar adjacent artworks are,
    revealing the curatorial rhythm and pacing of an auction catalogue.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with artwork features (one row per artwork)
    feature_cols : list of str
        Column names to use for similarity calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - folder_name: Auction lot/folder
        - position: Position in sequence
        - image1, image2: Consecutive images
        - similarity_score: 0-1 (1 = identical)
        
    Examples:
    ---------
    >>> # Assume df has color and spatial metrics
    >>> feature_cols = ['color_primary_pct', 'edge_density', 'symmetry_score']
    >>> similarities = calculate_pairwise_similarity(df, feature_cols)
    >>> print(similarities[['position', 'similarity_score']])
    """
    similarities = []
    
    # Group by folder (each folder is one auction sequence)
    for folder_name in df['folder_name'].unique():
        folder_df = df[df['folder_name'] == folder_name].sort_values('catalogue_position')
        
        for i in range(len(folder_df) - 1):
            vec1 = folder_df.iloc[i][feature_cols].values
            vec2 = folder_df.iloc[i+1][feature_cols].values
            
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                similarity = 0.0
            
            similarities.append({
                'folder_name': folder_name,
                'position': folder_df.iloc[i]['catalogue_position'],
                'image1': folder_df.iloc[i]['image_filename'],
                'image2': folder_df.iloc[i+1]['image_filename'],
                'similarity_score': round(similarity, 4)
            })
    
    return pd.DataFrame(similarities)


def detect_transitions(df: pd.DataFrame, 
                      feature_cols: List[str], 
                      threshold: float = 0.85) -> pd.DataFrame:
    """
    Detect curatorial transition points where visual style changes abruptly.
    
    A transition is detected when similarity drops below threshold,
    indicating a deliberate curatorial break (e.g., moving from abstract
    to figurative works, or from monochrome to colorful pieces).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with artwork features
    feature_cols : list of str
        Features to use for comparison
    threshold : float, default=0.85
        Similarity threshold (below = transition)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with transition points marked
        
    Examples:
    ---------
    >>> transitions = detect_transitions(df, feature_cols=['color_primary_pct'])
    >>> print(f"Found {len(transitions)} transition points")
    """
    sim_df = calculate_pairwise_similarity(df, feature_cols)
    
    # Mark transitions
    sim_df['is_transition'] = sim_df['similarity_score'] < threshold
    
    return sim_df[sim_df['is_transition']]


def identify_curatorial_blocks(df: pd.DataFrame, 
                               feature_cols: List[str],
                               threshold: float = 0.85) -> pd.DataFrame:
    """
    Segment artwork sequence into coherent curatorial blocks.
    
    Each block represents a group of visually similar consecutive artworks,
    separated by transition points. This reveals the curator's grouping strategy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with artwork features
    feature_cols : list of str
        Features for similarity calculation
    threshold : float, default=0.85
        Similarity threshold
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added 'block_id' column
        
    Examples:
    ---------
    >>> df_blocks = identify_curatorial_blocks(df, feature_cols)
    >>> # Count artworks per block
    >>> print(df_blocks.groupby(['folder_name', 'block_id']).size())
    """
    df_copy = df.copy()
    df_copy['block_id'] = 0
    
    # Calculate similarities
    sim_df = calculate_pairwise_similarity(df, feature_cols)
    
    for folder_name in df_copy['folder_name'].unique():
        folder_df = df_copy[df_copy['folder_name'] == folder_name].sort_values('catalogue_position')
        folder_sim = sim_df[sim_df['folder_name'] == folder_name]
        
        block_id = 0
        
        for idx in folder_df.index:
            df_copy.loc[idx, 'block_id'] = block_id
            
            # Check if this position is a transition
            position = df_copy.loc[idx, 'catalogue_position']
            transition_row = folder_sim[folder_sim['position'] == position]
            
            if not transition_row.empty and transition_row.iloc[0]['similarity_score'] < threshold:
                block_id += 1
    
    return df_copy


def calculate_visual_rhythm(df: pd.DataFrame, 
                            feature_cols: List[str],
                            window_size: int = 3) -> pd.DataFrame:
    """
    Calculate rolling visual rhythm (moving average of similarity scores).
    
    Smooth out noise to reveal overall pacing patterns in the catalogue.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with artwork features
    feature_cols : list of str
        Features for calculation
    window_size : int, default=3
        Window for rolling average
    
    Returns:
    --------
    pd.DataFrame
        Similarity scores with smoothed rhythm column
        
    Examples:
    ---------
    >>> rhythm = calculate_visual_rhythm(df, feature_cols, window_size=5)
    >>> print(rhythm[['position', 'similarity_score', 'rhythm_smoothed']])
    """
    sim_df = calculate_pairwise_similarity(df, feature_cols)
    
    # Calculate rolling average per folder
    sim_df['rhythm_smoothed'] = sim_df.groupby('folder_name')['similarity_score'].transform(
        lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean()
    )
    
    return sim_df


def summarize_sequence_statistics(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Generate summary statistics for each auction sequence.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with artwork features
    feature_cols : list of str
        Features to analyze
    
    Returns:
    --------
    pd.DataFrame
        Summary table with folder-level stats
        
    Examples:
    ---------
    >>> summary = summarize_sequence_statistics(df, feature_cols)
    >>> print(summary)
    """
    sim_df = calculate_pairwise_similarity(df, feature_cols)
    
    summary = sim_df.groupby('folder_name').agg({
        'similarity_score': ['mean', 'std', 'min', 'max'],
        'position': 'count'
    }).round(3)
    
    summary.columns = ['avg_similarity', 'std_similarity', 'min_similarity', 
                       'max_similarity', 'num_comparisons']
    
    return summary.reset_index()
