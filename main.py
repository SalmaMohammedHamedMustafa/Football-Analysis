import ultralytics
from my_utils import read_video, save_video
from tracker import Tracker

def main():
    tracker = Tracker(
        model_path="best.pt",
        confidence_threshold=0.1
    )

    processed_frames = tracker.process_video(
        input_path="input.mp4",
        output_path="output.mp4",
        use_existing_tracks=True,
        tracks_path="tracks.pkl"
    )

if __name__ == "__main__":
    main()