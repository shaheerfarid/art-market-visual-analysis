# Data Directory

## Structure

Organize your artwork images in folders, where each folder represents one auction lot or catalogue sequence:

```
data/
├── Auction_2023_Fall/
│   ├── artwork_001.jpg
│   ├── artwork_002.jpg
│   ├── artwork_003.jpg
│   └── ...
│
├── Auction_2024_Spring/
│   ├── artwork_001.jpg
│   ├── artwork_002.jpg
│   └── ...
│
└── YourCustomFolder/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Important Notes

1. **File Format**: Supported formats include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

2. **Naming Convention**: Files should be named to reflect their catalogue order (e.g., `001.jpg`, `002.jpg`, or `artwork_001.jpg`)

3. **Folder Names**: Each folder name will be used in analysis reports, so choose descriptive names

4. **File Ordering**: Files are processed in alphabetical order within each folder

## Image Requirements

- **Minimum Resolution**: 200x200 pixels (higher is better)
- **Maximum Size**: No hard limit, but images are resized to 400px max dimension for processing efficiency
- **Color Space**: RGB (RGBA images are converted automatically)
- **Quality**: Use original/high-quality scans for best analysis results

## Sample Dataset

If you don't have your own dataset, you can use:
- Public domain artworks from [WikiArt](https://www.wikiart.org/)
- Museum collections (check licensing)
- Your own photography of gallery exhibitions

## Privacy & Copyright

⚠️ **Important**: This repository does not include any artwork images to respect copyright. Users must provide their own images or use public domain sources.

The `data/` folder is gitignored by default to prevent accidental inclusion of proprietary artwork images.
