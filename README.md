# AI Lip Reading Project

A comprehensive AI-powered lip reading system that can transcribe speech from video by analyzing lip movements using computer vision and deep learning techniques.

## 🚀 Features

- **Real-time Lip Detection**: Uses MediaPipe for accurate facial landmark detection
- **Dual Model Support**: 
  - Basic (Offline) model for local processing
  - Premium (Cloud) model for enhanced accuracy
- **Multi-language Support**: Configurable language settings
- **Face Detection**: Optional face detection for improved lip tracking
- **Progress Tracking**: Real-time progress bars for video processing

## 📋 Requirements

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Dependencies
```
opencv-python>=4.5.0
torch>=1.9.0
numpy>=1.21.0
mediapipe>=0.8.0
tqdm>=4.62.0
deepface>=0.0.75
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/1hm544i5f1ll1/ai-lip-reading.git
   cd ai-lip-reading
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Basic Usage (Offline Model)
```bash
python lipreading.py --model basic --video_file path/to/your/video.mp4
```

### Premium Usage (Cloud Model)
```bash
python lipreading.py --model premium --video_file path/to/your/video.mp4 --api_key YOUR_API_KEY
```

### With Face Detection
```bash
python lipreading.py --model basic --video_file path/to/your/video.mp4 --detect_faces
```

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | Yes | Choose model type: `basic` or `premium` |
| `--video_file` | Yes | Path to the input video file |
| `--api_key` | For premium | API key for cloud-based model |
| `--language` | No | Language code (default: "en") |
| `--detect_faces` | No | Enable face detection preprocessing |

## 🔧 Technical Details

### Architecture
- **MediaPipe Integration**: For facial landmark detection and lip region extraction
- **Deep Learning Models**: 3D CNN + LSTM architecture for temporal feature extraction
- **Multi-processing**: Optimized for performance on multi-core systems

### Lip Detection Process
1. **Face Detection**: Locate faces in video frames
2. **Landmark Extraction**: Extract 61-88 facial landmarks (lip region)
3. **Region Cropping**: Crop and preprocess lip regions
4. **Model Inference**: Process through lip reading model
5. **Text Generation**: Generate transcription from lip movements

## 📊 Performance Metrics

- **Word Error Rate (WER)**: Measures transcription accuracy
- **Character Error Rate (CER)**: Character-level accuracy assessment
- **Processing Speed**: Frames per second (FPS) processing capability

## 🚧 Limitations

- **Model Dependencies**: Requires specific pre-trained models for optimal performance
- **Computational Resources**: GPU recommended for real-time processing
- **Video Quality**: Performance depends on video resolution and lighting conditions
- **Language Support**: Currently optimized for English (expandable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for facial landmark detection
- [PyTorch](https://pytorch.org/) for deep learning framework
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact: [your-email@example.com]

---

**Note**: This is a portfolio project demonstrating AI lip reading capabilities. For production use, additional model training and optimization may be required.
