"""
Semantic Analysis Module
=========================

CLIP-based zero-shot classification for art content, style, and medium detection.
"""

import torch
import clip
from PIL import Image
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# Global CLIP model (loaded once)
_model = None
_preprocess = None
_device = None
_text_features = None

# Comprehensive art taxonomy for CLIP prompts
CLIP_PROMPTS = {
    # Abstract categories
    'abstract_general': 'abstract art painting',
    'abstract_monochrome': 'monochromatic abstract painting with single color',
    'abstract_multicolor': 'colorful abstract art with multiple vibrant colors',
    'abstract_geometric': 'geometric abstract art with straight lines and shapes',
    'abstract_fluid': 'organic abstract art with flowing curves and fluid forms',

    # Figurative categories
    'figurative_general': 'figurative realistic art with recognizable subjects',
    'human_presence': 'painting with human figures or people',
    'human_male': 'portrait of a man or male figure in art',
    'human_female': 'portrait of a woman or female figure in art',
    'portrait_single': 'portrait painting of one single person',
    'portrait_group': 'group portrait with multiple people',
    'human_nude': 'artistic nude figure painting',
    'animal_presence': 'painting with animals or wildlife',

    # Landscape categories
    'landscape_general': 'landscape painting or outdoor scene',
    'landscape_natural': 'natural landscape with mountains forests or nature',
    'landscape_urban': 'urban landscape with buildings city or architecture',

    # Still life
    'still_life': 'still life painting with objects on table',

    # Medium categories
    'medium_painting': 'oil painting or acrylic painting with visible brushstrokes',
    'medium_sculpture': 'three dimensional sculpture or carved artwork',
    'medium_installation': 'contemporary art installation or environmental art',
    'medium_photograph': 'photograph or photographic artwork',
    'medium_design': 'design object or decorative applied art',
    'medium_print': 'print etching or lithograph artwork',
}


def initialize_clip_model(model_name: str = "ViT-B/32", device: Optional[str] = None):
    """
    Initialize CLIP model globally (call once at program start).
    
    Parameters:
    -----------
    model_name : str, default="ViT-B/32"
        CLIP model variant to use
    device : str, optional
        Device to use ('cuda', 'cpu', or None for auto-detect)
        
    Examples:
    ---------
    >>> initialize_clip_model()
    >>> # Now you can call analyze_with_clip() multiple times efficiently
    """
    global _model, _preprocess, _device, _text_features
    
    if device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device
    
    print(f"Loading CLIP model on {_device}...")
    _model, _preprocess = clip.load(model_name, device=_device)
    _model.eval()
    
    # Pre-compute text features for efficiency
    text_inputs = torch.cat([
        clip.tokenize(f"a photo of {prompt}")
        for prompt in CLIP_PROMPTS.values()
    ]).to(_device)
    
    with torch.no_grad():
        _text_features = _model.encode_text(text_inputs)
        _text_features /= _text_features.norm(dim=-1, keepdim=True)
    
    print("CLIP model loaded and text features cached!")


def analyze_with_clip(image_path: str) -> Dict:
    """
    Analyze artwork content using OpenAI CLIP for zero-shot classification.
    
    Returns confidence scores for 20+ art categories including:
    - Abstract vs. figurative
    - Geometric vs. fluid
    - Landscape types (natural/urban)
    - Human/animal presence
    - Medium detection (painting, sculpture, photo, etc.)
    
    Parameters:
    -----------
    image_path : str
        Path to the artwork image
    
    Returns:
    --------
    dict
        Dictionary mapping category names to confidence scores (0-1)
        Keys match CLIP_PROMPTS keys (e.g., 'abstract_general', 'human_presence')
        
    Examples:
    ---------
    >>> # First initialize the model
    >>> initialize_clip_model()
    >>> 
    >>> # Then analyze images
    >>> scores = analyze_with_clip("artwork.jpg")
    >>> print(f"Abstract: {scores['abstract_general']:.3f}")
    >>> print(f"Figurative: {scores['figurative_general']:.3f}")
    >>> 
    >>> # Determine medium
    >>> medium_scores = {k: v for k, v in scores.items() if k.startswith('medium_')}
    >>> best_medium = max(medium_scores, key=medium_scores.get)
    >>> print(f"Detected medium: {best_medium}")
    """
    global _model, _preprocess, _device, _text_features
    
    if _model is None:
        raise RuntimeError("CLIP model not initialized! Call initialize_clip_model() first.")
    
    # Load and preprocess image
    image = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(_device)

    # Calculate image features
    with torch.no_grad():
        image_features = _model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity with text prompts
        similarity = (100.0 * image_features @ _text_features.T).softmax(dim=-1)

    # Map to dictionary
    results = {}
    for i, key in enumerate(CLIP_PROMPTS.keys()):
        results[key] = round(similarity[0, i].item(), 4)

    return results


def get_dominant_medium(clip_scores: Dict) -> str:
    """
    Extract the dominant medium from CLIP scores.
    
    Parameters:
    -----------
    clip_scores : dict
        Dictionary returned by analyze_with_clip()
    
    Returns:
    --------
    str
        Dominant medium ('painting', 'sculpture', 'photograph', etc.)
        
    Examples:
    ---------
    >>> scores = analyze_with_clip("artwork.jpg")
    >>> medium = get_dominant_medium(scores)
    >>> print(f"This artwork is a {medium}")
    """
    medium_scores = {k: v for k, v in clip_scores.items() if k.startswith('medium_')}
    if not medium_scores:
        return 'unknown'
    
    dominant = max(medium_scores, key=medium_scores.get)
    return dominant.replace('medium_', '')


def get_dominant_style(clip_scores: Dict) -> str:
    """
    Extract the dominant style (abstract, figurative, landscape, still life).
    
    Parameters:
    -----------
    clip_scores : dict
        Dictionary returned by analyze_with_clip()
    
    Returns:
    --------
    str
        Dominant style category
        
    Examples:
    ---------
    >>> scores = analyze_with_clip("artwork.jpg")
    >>> style = get_dominant_style(scores)
    >>> print(f"Style: {style}")
    """
    style_categories = {
        'abstract': clip_scores.get('abstract_general', 0),
        'figurative': clip_scores.get('figurative_general', 0),
        'landscape': clip_scores.get('landscape_general', 0),
        'still_life': clip_scores.get('still_life', 0)
    }
    
    return max(style_categories, key=style_categories.get)
