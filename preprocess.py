import cv2
import os
from moviepy import VideoFileClip

def run_preprocessing(video_path, frames_folder, audio_path):
    #Create folder for frames
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    #Extract Audio (for YAMNet model)
    video_clip = VideoFileClip(video_path)
    if video_clip.audio:
        video_clip.audio.write_audiofile(audio_path)
    video_clip.close()

    #Extract Frames (for I3D model)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save one frame every 30 frames (normalization)
        if count % 30 == 0:
            frame_name = os.path.join(frames_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        count += 1

    cap.release()
    print(f"Done: {saved_count} frames and audio extracted.")

if __name__ == "__main__":
    # Settings
    VIDEO_IN = 'videos/test_video.mp4'
    FRAMES_OUT = 'data/extracted_frames'
    AUDIO_OUT = 'data/test_audio.wav'
    
    run_preprocessing(VIDEO_IN, FRAMES_OUT, AUDIO_OUT)