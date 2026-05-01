import torch
import torch.nn as nn
import librosa
import numpy as np
import os

class AudioViolenceModel:
    def __init__(self):
        print("Loading PyTorch-based Audio Model...")
        # כאן בעתיד נטען קובץ .pt אמיתי
        # self.model = torch.load('trained_model/audio_model.pt')

    def predict(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)
        
        # ניתוח תדרים (Mel-Spectrogram) במקום רק עוצמה
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # חישוב סטיית תקן - באלימות (צעקות/נפצים) יש שינויי תדרים חדים
        audio_score = np.clip(np.std(log_S) / 20.0, 0.1, 0.9)
        print(f"Audio Frequency Analysis Score: {audio_score:.2f}")
        return float(audio_score)

class VideoViolenceModel:
    def __init__(self):
        print("Loading PyTorch Video I3D Model...")
        # הכנה לטעינת משקולות
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, frames_path):
        print(f"Analyzing frames in: {frames_path}")
        
        # ספירת כמות הפריימים שחולצו
        num_frames = len([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
        
        # סימולציה של ניתוח תנועה: בסרטוני אלימות בדר"כ יש יותר תנועה (פריימים)
        # במערכת סופית כאן נכניס את ה-Tensor למודל I3D
        video_score = np.clip(num_frames / 100.0, 0.2, 0.85)
        return float(video_score)