# Art Market Visual Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-orange.svg)](https://github.com/openai/CLIP)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade computer vision pipeline for analyzing artwork patterns, color theory, and curatorial sequencing in auction catalogues using deep learning and traditional CV techniques.**

---

## Project Overview

This system bridges art market analytics with state-of-the-art computer vision, enabling **quantitative analysis of visual patterns** in auction house catalogues. By combining traditional image processing with OpenAI's CLIP model, it extracts actionable insights from artwork sequences that inform curatorial strategy, market positioning, and art historical research.

### **The Problem**

Auction houses organize hundreds of artworks in carefully curated sequences. Understanding the visual logic behind these arrangements—color progression, style clustering, thematic transitions—has traditionally been qualitative and labor-intensive. This project **automates and quantifies** that analysis.

### **The Solution**

A modular, scalable pipeline that:
- **Extracts 40+ visual metrics** from each artwork (color theory, geometry, composition)
- **Classifies content semantically** using CLIP (abstract vs. figurative, geometric vs. organic, medium types)
- **Detects curatorial patterns** by analyzing sequential similarity and transition points
- **Generates publication-ready reports** with statistical tables and comparative visualizations

---

## Key Features

| Feature | Description | Technology |
|---------|-------------|------------|
| **Color Theory Analysis** | Primary/secondary/tertiary color distributions with HSV clustering | OpenCV + K-Means |
| **Semantic Classification** | Zero-shot art style and content detection (abstract, figurative, landscape, etc.) | OpenAI CLIP (ViT-B/32) |
| **Spatial Analysis** | Symmetry detection, edge orientation, geometric shape recognition | OpenCV Canny + Hough Transform |
| **Sequence Analysis** | Cosine similarity between consecutive artworks to detect curatorial blocks | NumPy + SciPy |
| **Multi-Folder Comparison** | Cross-catalogue analysis with heatmaps, radar charts, and statistical tests | Matplotlib + Plotly + Seaborn |
| **Automated Reporting** | Excel exports with 9+ statistical summary tables for academic presentation | Pandas + openpyxl |

---

## 📊 Sample Results

### Color Theory Distribution
```
Folder/Lot         | Primary Colors | Secondary Colors | Tertiary Colors
-------------------|----------------|------------------|----------------
Auction_2023_Fall  | 34.2%         | 28.7%           | 37.1%
Auction_2024_Spring| 41.5%         | 25.3%           | 33.2%
```

### Content Classification (CLIP Confidence Scores)
```
Abstract: 0.652 | Figurative: 0.234 | Landscape: 0.089 | Still Life: 0.025
```

### Detected Visual Clusters
```
✓ Cluster 1 (Positions 1-15): Monochrome geometric abstracts (avg similarity: 0.87)
✓ Cluster 2 (Positions 16-34): Figurative portraits (avg similarity: 0.82)
✓ Transition Point: Position 35 (similarity drop: 0.41)
```

---

## Tech Stack

### **Core Technologies**
- **Python 3.8+** - Primary language
- **OpenCV** - Traditional computer vision (edge detection, color analysis, shape recognition)
- **OpenAI CLIP** - Transformer-based semantic understanding of artwork content
- **PyTorch** - Deep learning framework for CLIP inference
- **scikit-learn** - K-Means clustering for color quantization
- **Pandas & NumPy** - Data manipulation and statistical analysis

### **Visualization & Reporting**
- **Matplotlib** - Statistical visualizations
- **Seaborn** - Heatmaps and correlation plots
- **Plotly** - Interactive radar charts and dashboards
- **openpyxl** - Professional Excel report generation

---

## 📁 Project Structure

```
art-market-visual-analysis/
├── README.md                          # You are here
├── requirements.txt                   # Dependency management
├── .gitignore                         # Exclude data and outputs
│
├── src/                               # Modular source code
│   ├── color_analysis.py             # HSV color extraction & clustering
│   ├── spatial_analysis.py           # Geometry, symmetry, edge detection
│   ├── semantic_analysis.py          # CLIP-based content classification
│   ├── sequence_analysis.py          # Temporal pattern & transition detection
│   └── utils.py                      # Image loading & preprocessing
│
├── data/                              # Your image datasets (gitignored)
│   └── README.md                      # Data structure guide
│
├── outputs/                           # Results (gitignored)
│   ├── reports/                       # Excel statistical summaries
│   ├── visualizations/                # PNG/HTML charts
│   └── logs/                          # Processing logs
│
├── docs/                              # Technical documentation
│   ├── methodology.md                 # Algorithm explanations
│   └── metrics_reference.md          # All 40+ metrics defined
│
└── examples/                          # Quick start scripts
    └── analyze_single_folder.py
```

---

## Quick Start

### **1. Installation**

```bash
# Clone the repository
git clone https://github.com/shaheerfarid/art-market-visual-analysis.git
cd art-market-visual-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### **2. Prepare Your Data**

Organize your artwork images in folders (each folder represents one auction lot/sequence):

```
data/
├── Auction_2023_Fall/
│   ├── artwork_001.jpg
│   ├── artwork_002.jpg
│   └── ...
├── Auction_2024_Spring/
│   ├── artwork_001.jpg
│   └── ...
```

### **3. Run Analysis**

#### **Option A: Command Line**
```bash
python examples/analyze_single_folder.py --input data/Auction_2023_Fall --output outputs/results.xlsx
```

#### **Option B: Jupyter Notebook**
```bash
jupyter notebook notebooks/02_complete_pipeline.ipynb
```

#### **Option C: Python Script**
```python
from src.color_analysis import extract_color_metrics
from src.semantic_analysis import analyze_with_clip
from src.sequence_analysis import detect_visual_clusters

# Analyze single image
color_data = extract_color_metrics("data/Auction_2023_Fall/artwork_001.jpg")
semantic_data = analyze_with_clip("data/Auction_2023_Fall/artwork_001.jpg")

# Analyze entire folder
clusters = detect_visual_clusters("data/Auction_2023_Fall/")
```

---

## How It Works

### **Pipeline Architecture**

1. **Image Loading** - RGB conversion, resize for efficiency
2. **Color Extraction** - K-Means clustering (8 clusters) → HSV categorization
3. **Spatial Features** - Edge density, symmetry score, shape detection
4. **Semantic Classification** - CLIP embedding → 20+ art style categories
5. **Sequence Analysis** - Cosine similarity between consecutive works
6. **Cluster Detection** - Threshold-based segmentation (default: 0.85 similarity)
7. **Report Generation** - 9 Excel sheets + 7 visualization types

---

## Metrics Explained

### **Color Metrics (12 features)**
- **Dominant Colors**: Blue, Yellow, Red, Green, Orange, Purple, White, Black (% coverage)
- **Color Theory**: Primary, Secondary, Tertiary color percentages
- **Clustering**: Number of distinct color clusters (K=8)

### **Spatial Metrics (8 features)**
- **Symmetry Score**: Left-right correlation (0-1)
- **Edge Density**: Canny edge pixel ratio
- **Orientation**: Vertical vs. horizontal mass distribution
- **Geometry Confidence**: Circle, rectangle, square, triangle detection (0-1)

### **Semantic Metrics (20+ features)**
- **Style Categories**: Abstract (general, monochrome, multicolor, geometric, fluid)
- **Content Types**: Figurative, landscape (natural/urban), still life, portrait
- **Subject Matter**: Human presence, animal presence, gender classification
- **Medium Detection**: Painting, sculpture, photograph, print, installation

### **Sequence Metrics (3 features)**
- **Visual Similarity**: Cosine similarity between feature vectors
- **Semantic Distance**: Euclidean distance in CLIP embedding space
- **Transition Detection**: Binary flag for curatorial breaks (similarity < threshold)

---

## Output Examples

### **1. Statistical Summary Tables** 
9 Excel sheets covering color theory, content classification, medium distribution, spatial organization, and geometry detection.

### **2. Visualizations**
- Color theory comparison (bar charts)
- Color distribution heatmaps
- Content type comparison (grouped bars)
- Medium distribution (stacked bars)
- Spatial metrics comparison
- Interactive radar charts (HTML)

### **3. Sequence Analysis**
Pairwise similarity scores showing visual transitions and curatorial blocks.

---

## Use Cases

### **For Auction Houses**
- **Catalogue Optimization**: Identify visual rhythm and pacing in lot sequences
- **Market Positioning**: Compare color/style distributions across competing sales
- **Trend Analysis**: Track evolution of abstract vs. figurative ratios over time

### **For Art Historians**
- **Curatorial Strategy**: Quantify how major museums organize exhibitions
- **Pattern Recognition**: Detect subtle thematic clusters in large archives
- **Comparative Studies**: Statistical analysis of color theory across art movements

### **For Data Scientists**
- **Computer Vision Portfolio**: Demonstrates OpenCV + Deep Learning integration
- **Production ML**: Shows model deployment (CLIP) with preprocessing pipeline
- **Data Engineering**: ETL pipeline for unstructured visual data → structured analytics

---
