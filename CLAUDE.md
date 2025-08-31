# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAI learning repository containing example implementations for different types of machine learning tasks using the FastAI framework. The repository includes practical examples for computer vision, natural language processing, tabular data analysis, and image segmentation.

## Repository Structure

The codebase consists of standalone Python scripts, each demonstrating different FastAI capabilities:

- `imgmodel.py` - Image classification using pets dataset (cats vs dogs)
- `bearimgmodel.py` - Bear image classification with RapidAPI image search integration  
- `camvidmodel.py` - Image segmentation using CAMVID dataset with U-Net architecture
- `imdbmodel.py` - Text classification for movie reviews using IMDB dataset
- `csvmodel.py` - Tabular data classification using Adult Census dataset
- `images/` - Directory containing sample images for testing models
- `FastAi.txt` - Learning notes with vocabulary definitions and helpful links

## Common Development Commands

Since this is a collection of learning scripts, each file is meant to be run independently:

```bash
# Run individual models
python imgmodel.py          # Pet classification
python bearimgmodel.py      # Bear image search and classification  
python camvidmodel.py       # Image segmentation
python imdbmodel.py         # Text sentiment analysis
python csvmodel.py          # Tabular data classification
```

## Architecture Patterns

### Data Loading Patterns
- **Vision**: Uses `ImageDataLoaders.from_name_func()` with custom labeling functions
- **Segmentation**: Uses `SegmentationDataLoaders.from_label_func()` with path-based label mapping
- **Text**: Uses `TextDataLoaders.from_folder()` for folder-based text classification
- **Tabular**: Uses `TabularDataLoaders.from_csv()` with preprocessing pipelines

### Model Training Patterns  
- All models check for CUDA availability and report GPU usage
- Vision models use `vision_learner()` with pre-trained ResNet architectures
- Segmentation uses `unet_learner()` for pixel-wise classification
- Text uses `text_classifier_learner()` with AWD-LSTM architecture
- Tabular uses `tabular_learner()` with automated feature processing

### Fine-tuning Strategy
- Vision models: `learn.fine_tune(1)` for quick transfer learning
- Segmentation: `learn.fine_tune(8)` for more complex spatial learning
- Text: `learn.fine_tune(4, 1e-2)` with custom learning rate
- Tabular: `learn.fit_one_cycle(3)` using one-cycle training policy

## Key Dependencies

- `fastai` - Main deep learning framework
- `torch` - PyTorch backend for GPU acceleration
- `requests` - For API-based image search (bearimgmodel.py)
- `sklearn` - Additional metrics and evaluation tools (csvmodel.py)

## Data Sources

Models automatically download datasets through FastAI's URL system:
- `URLs.PETS` - Oxford-IIIT Pet Dataset for image classification
- `URLs.CAMVID_TINY` - Cambridge-driving Labeled Video Database for segmentation  
- `URLs.IMDB` - Movie review dataset for sentiment analysis
- `URLs.ADULT_SAMPLE` - Census income dataset for tabular classification

Local data path for cached datasets: `C:\Users\mac-r\.fastai\data\`

## API Integration

`bearimgmodel.py` includes RapidAPI integration for real-time image search. The API key is hardcoded for learning purposes but should be moved to environment variables in production use.