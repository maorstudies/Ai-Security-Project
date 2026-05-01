import os
from preprocess import run_preprocessing
from model import VideoViolenceModel, AudioViolenceModel

def main():
    # הנתיב לתיקיית הבדיקות שסידרת
    base_test_path = 'videos/test_set'
    categories = ['violence', 'non_violence']
    
    # אתחול המודלים
    v_model = VideoViolenceModel()
    a_model = AudioViolenceModel()
    
    print(f"--- Starting Batch Test on {base_test_path} ---")
    
    for category in categories:
        folder_path = os.path.join(base_test_path, category)
        # רשימת כל הסרטונים בתיקייה
        video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            print(f"\nTesting: {video_file} (Category: {category})")
            
            # שלב 1: עיבוד (חילוץ פריימים ואודיו לתיקיות זמניות)
            frames_dir = 'data/extracted_frames'
            audio_file = 'data/test_audio.wav'
            run_preprocessing(video_path, frames_dir, audio_file)
            
            # שלב 2: חיזוי
            v_score = v_model.predict(frames_dir)
            a_score = a_model.predict(audio_file)
            
            # שלב 3: החלטה (Late Fusion)
            final_score = (v_score + a_score) / 2
            result = "VIOLENCE" if final_score > 0.5 else "NORMAL"
            print(f"Result: {result} (Score: {final_score:.2f})")

if __name__ == "__main__":
    main()