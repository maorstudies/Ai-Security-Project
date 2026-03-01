from preprocess import run_preprocessing
from model import VideoViolenceModel, AudioViolenceModel

def main():
    # Paths defined in project architecture
    video_file = 'videos/test_video.mp4'
    frames_dir = 'data/extracted_frames'
    audio_file = 'data/test_audio.wav'

    # Step 1: Preprocessing (Extract frames and audio)
    print("--- Step 1: Preprocessing ---")
    run_preprocessing(video_file, frames_dir, audio_file)

    # Step 2: Inference (Get predictions from models)
    print("\n--- Step 2: Model Inference ---")
    video_model = VideoViolenceModel()
    audio_model = AudioViolenceModel()

    v_score = video_model.predict(frames_dir)
    a_score = audio_model.predict(audio_file)

    # Step 3: Decision Logic (Fusion)
    # Combine both results to decide if violence is present
    print("\n--- Step 3: Final Decision ---")
    final_score = (v_score + a_score) / 2
    
    if final_score > 0.5:
        print("RESULT: VIOLENCE DETECTED (YES)")
    else:
        print("RESULT: NO VIOLENCE (NO)") [cite: 7, 150]

if __name__ == "__main__":
    main()