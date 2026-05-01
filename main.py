import os
from preprocess import run_preprocessing
from model import VideoViolenceModel, AudioViolenceModel

def main():
    # נתיבי הפרויקט
    base_test_path = 'videos/test_set'
    categories = ['violence', 'non_violence']
    
    # אתחול המודלים (ייטענו מה-Cache שכבר יצרת)
    v_model = VideoViolenceModel()
    a_model = AudioViolenceModel()
    
    print(f"\n--- Starting Batch Test on {base_test_path} ---")
    
    for category in categories:
        folder_path = os.path.join(base_test_path, category)
        if not os.path.exists(folder_path): continue
        
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            print(f"\nTesting: {video_file} (Category: {category})")
            
            # שלב 1: עיבוד (חילוץ נקי של פריימים ואודיו)
            frames_dir = 'data/extracted_frames'
            audio_file = 'data/test_audio.wav'
            run_preprocessing(video_path, frames_dir, audio_file)
            
            # שלב 2: חיזוי מבוסס AI
            v_score = v_model.predict(frames_dir)
            a_score = a_model.predict(audio_file)
            
            # שלב 3: החלטה סופית (Late Fusion)
            # נותנים עדיפות לוידאו (0.7) כדי להוריד רגישות יתר של האודיו
            final_score = (v_score * 0.7) + (a_score * 0.3)
            
            result = "VIOLENCE" if final_score > 0.70 else "NORMAL"
            print(f"Final Decision: {result} (Score: {final_score:.2f})")

if __name__ == "__main__":
    main()