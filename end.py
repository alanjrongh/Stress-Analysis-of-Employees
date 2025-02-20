import cv2
import torch
import numpy as np
import pyaudio
import librosa
import wave
import threading
import tkinter as tk
from tkinter import Label, ttk
from pygame import mixer
import time
import io
from array import array
from deepface import DeepFace
from torchvision import transforms
from cmpltd_speech import ImprovedAudioEmotionModel, AudioPreprocessor

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained speech emotion model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_model = ImprovedAudioEmotionModel().to(device)
audio_model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
audio_model.eval()
preprocessor = AudioPreprocessor()

# Constants and mappings
LABEL_MAP = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'neutral'}
STRESS_SUGGESTIONS = {
    'anger': "Take deep breaths and relax.",
    'disgust': "Try to shift focus to something positive.",
    'fear': "Take a short walk and clear your mind.",
    'sad': "Listen to some calming music or talk to a friend.",
    'neutral': "Keep up the good work!",
    'happy': "Maintain your positive energy!"
}
STRESS_LEVELS = {
    'anger': 80,
    'disgust': 70,
    'fear': 90,
    'sad': 75,
    'neutral': 20,
    'happy': 10
}

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

# Global variables for thread synchronization
frame_count = 0
current_audio_emotion = "Unknown"
current_face_emotion = "Unknown"
emotion_lock = threading.Lock()
stop_flag = False

class StressAnalysisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Worker Stress Analysis")
        self.root.geometry("600x550")
        self.root.attributes('-topmost', True)  # Set Tkinter window to be topmost
        
        # Initialize sound handling
        mixer.init()
        self.prepare_alert_sound()
        self.last_alert_time = 0
        self.alert_cooldown = 3  # Minimum seconds between alerts
        
        # Create GUI elements
        self.label_face = Label(self.root, text="Face Emotion: Detecting...", font=("Arial", 14))
        self.label_face.pack(pady=10)
        
        self.label_audio = Label(self.root, text="Audio Emotion: Detecting...", font=("Arial", 14))
        self.label_audio.pack(pady=10)
        
        self.label_suggestion = Label(self.root, text="Suggestion: ...", 
                                    font=("Arial", 14), wraplength=500)
        self.label_suggestion.pack(pady=10)
        
        self.stress_bar = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.stress_bar.pack(pady=10)
        
        self.stress_label = Label(self.root, text="Stress Level: 0%", font=("Arial", 14))
        self.stress_label.pack(pady=10)
        
        # Start the processing threads
        threading.Thread(target=self.update_ui, daemon=True).start()
        threading.Thread(target=self.audio_processing_loop, daemon=True).start()

    def prepare_alert_sound(self):
        """Create a simple beep sound"""
        freq = 440  # frequency in Hz
        duration = 0.5  # duration in seconds
        sample_rate = 44100  # sample rate in Hz
        
        # Generate samples
        samples = array('h')
        period = int(sample_rate / freq)
        for i in range(int(duration * sample_rate)):
            value = 32767 * float(i % period) / period
            samples.append(int(value))
        
        # Create wave file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(samples.tobytes())
        
        buffer.seek(0)
        self.alert_sound = mixer.Sound(buffer)

    def play_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.alert_sound.play()
            self.last_alert_time = current_time

    def capture_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                       input=True, frames_per_buffer=CHUNK)
        frames = []
        
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def predict_audio_emotion(self):
        mel_spec_db = preprocessor.process_audio(WAVE_OUTPUT_FILENAME)
        if mel_spec_db is None:
            return "Unknown"
        
        mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-8)
        input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = audio_model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return LABEL_MAP[prediction]

    def process_frame(self, frame):
        """Process a single frame for face emotion detection"""
        global current_face_emotion, frame_count
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, 
                                            minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            # Reduce DeepFace analysis frequency
            if frame_count % 10 == 0:  # Only analyze every 10th frame
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], 
                                            enforce_detection=False)
                    current_face_emotion = result[0]['dominant_emotion']
                except Exception as e:
                    print(f"Face analysis error: {e}")
                    
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, current_face_emotion, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return frame

    def audio_processing_loop(self):
        """Separate thread for audio processing"""
        global current_audio_emotion
        
        while not stop_flag:
            try:
                self.capture_audio()
                audio_emotion = self.predict_audio_emotion()
                with emotion_lock:
                    current_audio_emotion = audio_emotion
                time.sleep(3)  # Wait before next audio processing
            except Exception as e:
                print(f"Audio processing error: {e}")
                time.sleep(1)

    def update_ui(self):
        """Main UI update loop"""
        global frame_count, current_audio_emotion, current_face_emotion
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create named window and set it to be topmost
        cv2.namedWindow('Worker Stress Analysis', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Worker Stress Analysis', cv2.WND_PROP_TOPMOST, 1)
        
        last_ui_update = time.time()
        
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            processed_frame = self.process_frame(frame)
            
            # Update UI at a lower frequency (every 500ms)
            current_time = time.time()
            if current_time - last_ui_update >= 0.5:
                with emotion_lock:
                    audio_emotion = current_audio_emotion
                    face_emotion = current_face_emotion
                    
                self.label_face.config(text=f"Face Emotion: {face_emotion}")
                self.label_audio.config(text=f"Audio Emotion: {audio_emotion}")
                
                detected_emotion = face_emotion if face_emotion != "Unknown" else audio_emotion
                suggestion = STRESS_SUGGESTIONS.get(detected_emotion, "Keep up the great work!")
                self.label_suggestion.config(text=f"Suggestion: {suggestion}")
                
                stress_level = STRESS_LEVELS.get(detected_emotion, 50)
                self.stress_bar['value'] = stress_level
                self.stress_label.config(text=f"Stress Level: {stress_level}%")
                
                if stress_level >= 80:
                    self.play_alert()
                    
                last_ui_update = current_time
            
            cv2.imshow('Worker Stress Analysis', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        finally:
            global stop_flag
            stop_flag = True

if __name__ == "__main__":
    app = StressAnalysisGUI()
    app.run()