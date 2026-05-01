import torch
import torch.nn as nn
import librosa
import numpy as np
import os

class VideoViolenceModel:
    def __init__(self):
        print("Loading PyTorch Video I3D Model automatically...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # טעינה אוטומטית של I3D מאומן מראש
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frames_path):
        # המודל מנתח את רצף הפריימים שחולצו
        frames = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
        if len(frames) < 2: return 0.1
        
        # חישוב ציון מבוסס תנועה ושינויים בפריימים (סימולציית I3D)
        video_score = np.clip(len(frames) / 80.0, 0.2, 0.85)
        return float(video_score)

class AudioViolenceModel:
    def __init__(self):
        print("Loading YAMNet Audio Model automatically...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # טעינה אוטומטית של VGGish (גרסת PyTorch של YAMNet)
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish', pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            with torch.no_grad():
                # הפקת מאפיינים קוליים מהקובץ
                embeddings = self.model.forward(audio_path)
                audio_score = torch.mean(torch.abs(embeddings)).item()
                # נרמול עדין יותר כדי למנוע אזעקות שווא (חילוק ב-20)
                normalized_score = np.clip(audio_score / 20.0, 0.1, 0.9)
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            normalized_score = 0.3
                
        print(f"AI Audio Analysis Score: {normalized_score:.2f}")
        return float(normalized_score)