# Real-Time Worker Stress Analysis using Face & Speech Emotion Recognition

This project analyzes real-time stress levels in workers using **facial expressions** and **speech emotion recognition**. It combines **computer vision** and **deep learning** to detect emotions and provide stress-related insights.

## Features
- **Real-time facial emotion detection** using OpenCV and DeepFace.
- **Audio-based emotion recognition** using a CNN model in PyTorch.
- **Multi-modal analysis** combining face and speech for stress evaluation.
- **Live visualization** with stress level indicators in a GUI.
- **Automated stress suggestions** based on detected emotions.

## Technologies Used
- **Computer Vision:** OpenCV, DeepFace
- **Audio Processing:** Librosa, PyTorch
- **Machine Learning:** CNN for speech emotion classification
- **GUI:** Tkinter for visualization
- **Multithreading:** Ensures smooth real-time performance

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository

2. Run the stress analysis system:
    ```sh
    python end.py

## How It Works
### Face Emotion Detection:
   - Uses **OpenCV** to capture real-time video.
   - **DeepFace** analyzes facial expressions.
### Speech Emotion Recognition:
   - Captures **3-second audio** clips.
   - Extracts **Mel Spectrogram** features.
   - Passes them to a **pre-trained CNN** model.
### Stress Analysis & Visualization:
   - Combines **facial and audio** emotions.
   - Displays **stress level** in a **GUI**.
   - Provides **stress-relief suggestions**.
## Usage
  - Press **q** to exit the face emotion detection window.
  - Ensure a working microphone for speech analysis.
  - The GUI updates in real-time based on detected emotions.
## Future Improvements
  - Adding stress trend analysis over time.
  - Expanding dataset for improved accuracy.
  - Integrating with wearable sensors for physiological stress data.
