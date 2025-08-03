import ultralytics
from my_utils import read_video, save_video
from team_assigner import TeamAssigner

from tracker import Tracker

def main():
    """Main function to run the tracker with improved ball detection."""
    tracker = Tracker(
        model_path="best.pt",
        confidence_threshold=0.02  # Updated to use lower threshold for better ball detection
    )
    
    print("Starting video processing with improved ball detection...")
    print(f"Ball detection threshold: {tracker.class_thresholds[0]}")
    
    processed_frames = tracker.process_video(
        input_path='input.mp4',
        output_path='output_with_camera_movement.mp4',
        tracks_path='tracks.pkl',
        camera_movement_stub_path='camera_movement.pkl',  # Will save/load camera movement data
        use_camera_movement=True,  # Enable camera movement compensation
        show_camera_movement=True  # Show camera movement overlay
    )
    
    print(f"\n✅ Video processing completed!")
    print(f"Total frames processed: {processed_frames}")
    print(f"Output video saved as: output.mp4")

if __name__ == "__main__":
    main()