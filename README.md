# 🐻 FastAI Learning Repository

A comprehensive collection of FastAI examples demonstrating various machine learning tasks including computer vision, natural language processing, tabular data analysis, and image segmentation.

## 📋 Overview

This repository contains practical implementations of different ML tasks using the FastAI framework:

- **🖼️ Image Classification** - Pet classification and bear type recognition
- **🎭 Image Segmentation** - Pixel-wise classification using U-Net
- **📝 Text Classification** - Movie review sentiment analysis  
- **📊 Tabular Data** - Census income prediction
- **🌐 Web Interface** - Interactive bear classifier with upload functionality

## 🚀 Quick Start

### Prerequisites
```bash
pip install fastai torch torchvision flask
```

### Running Individual Models

```bash
# Image classification (pets dataset)
python imgmodel.py

# Bear classification with API integration
python bearimgmodel.py

# Image segmentation
python camvidmodel.py

# Text sentiment analysis
python imdbmodel.py

# Tabular data classification
python csvmodel.py
```

### Web Interface

Launch the interactive bear classifier:

```bash
cd webapp
python web_app.py
```

Visit `http://localhost:5000` to upload and classify bear images.

## 📁 Project Structure

```
fastai/
├── 🐾 imgmodel.py          # Pet classification (cats vs dogs)
├── 🐻 bearimgmodel.py      # Bear type classification + API
├── 🎯 camvidmodel.py       # Image segmentation (CAMVID)
├── 🎬 imdbmodel.py         # Text classification (IMDB reviews)
├── 📈 csvmodel.py          # Tabular data (Adult Census)
├── 🌐 webapp/              # Web interface
│   ├── web_app.py          # Flask application
│   └── templates/index.html # Web UI
├── 🖼️ images/              # Training images (bears dataset)
├── 📊 outcome/             # Model analysis outputs
└── 📚 FastAi.txt           # Learning notes and resources
```

## 🎯 Model Highlights

### Bear Classifier
- **Architecture**: ResNet-18 with transfer learning
- **Classes**: Grizzly, Black, Teddy bears
- **Features**: Data augmentation, confusion matrix analysis, top losses visualization
- **Web Interface**: Upload and classify images instantly

### Pet Classifier  
- **Dataset**: Oxford-IIIT Pet Dataset
- **Task**: Binary classification (cats vs dogs)
- **Architecture**: ResNet-34

### Image Segmentation
- **Dataset**: CAMVID (road scene understanding)
- **Architecture**: U-Net with ResNet-34 backbone
- **Task**: Pixel-wise semantic segmentation

### Text Classification
- **Dataset**: IMDB movie reviews
- **Architecture**: AWD-LSTM 
- **Task**: Sentiment analysis (positive/negative)

### Tabular Classification
- **Dataset**: Adult Census Income
- **Task**: Income prediction (>=50k vs <50k)
- **Features**: Automated preprocessing pipeline

## 🔧 Key Features

### Terminal-Friendly Output
All models include terminal-compatible output for:
- Training progress visualization
- Model performance metrics
- Prediction results with confidence scores
- Analysis charts saved as image files

### GPU Support
Automatic GPU detection and utilization when available, with fallback to CPU.

### Model Export
Trained models are exported as `.pkl` files for easy deployment and inference.

### Data Augmentation
Comprehensive data augmentation strategies including:
- Random resized crops
- Padding and squishing
- Color and brightness adjustments

## 📊 Results

### Bear Classifier Performance
- **Validation Accuracy**: >95%
- **Training Time**: ~4 epochs (few minutes on GPU)
- **Model Size**: ~45MB

### Analysis Tools
- Confusion matrices
- Top loss analysis
- Prediction confidence scores
- Interactive error analysis

## 🛠️ Development

### Adding New Models
Follow the established patterns:
1. Use FastAI's high-level APIs
2. Include terminal-friendly output
3. Add GPU/CPU compatibility
4. Export trained models
5. Set `num_workers=0` for Windows compatibility

### API Integration
The bear model includes RapidAPI integration for real-time image search and dataset expansion.

## 📚 Learning Resources

- **FastAI Documentation**: https://docs.fast.ai/
- **FastAI Book**: Chapter examples and exercises
- **Dataset Sources**: Oxford VGG, UCI ML Repository
- **Pretrained Models**: ImageNet, various domain-specific models

## 🚀 Deployment

### Web Application
The Flask web app provides a production-ready interface for the bear classifier with:
- File upload support
- Real-time predictions
- Confidence scores
- Mobile-friendly design

### Model Serving
Exported models can be easily integrated into:
- REST APIs
- Mobile applications
- Batch processing pipelines
- Edge devices

## 📈 Future Enhancements

- [ ] Add more animal species to classifier
- [ ] Implement model versioning
- [ ] Add REST API endpoints
- [ ] Include model performance monitoring
- [ ] Add data drift detection

## 🤝 Contributing

This is a learning repository. Feel free to:
- Add new model examples
- Improve existing implementations
- Enhance the web interface
- Add more comprehensive testing

## 📄 License

Educational use - based on FastAI framework and examples.

---

**Built with ❤️ using FastAI and PyTorch**