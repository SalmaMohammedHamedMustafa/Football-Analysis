import os
import cv2
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import ultralytics
from ultralytics import YOLO
import sys
sys.path.append('../')
from my_utils import get_center_of_bbox, get_bbox_width


class Tracker:
    """
    Memory-efficient video tracker that processes videos in streaming mode.
    Handles object detection, tracking, and annotation rendering without
    loading entire videos into memory.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.1):
        """
        Initialize the Tracker.
        
        Args:
            model_path: Path to the YOLO model weights file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(str(self.model_path))
            print(f" Loaded model: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _get_video_info(self, video_path: Path) -> Tuple[int, int, int, float]:
        """
        Get video properties without loading the entire video.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Tuple of (width, height, total_frames, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        return width, height, total_frames, fps
    
    def _setup_video_writer(self, output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
        """
        Setup video writer for output.
        
        Args:
            output_path: Path for output video
            width: Frame width
            height: Frame height
            fps: Frames per second
            
        Returns:
            Configured VideoWriter object
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use mp4v codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for: {output_path}")
        
        return writer
    
    def _generate_tracks(self, video_path: Path, tracks_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate object tracks for the entire video.
        
        Args:
            video_path: Path to input video
            tracks_path: Optional path to save/load tracks
            
        Returns:
            Dictionary containing track data
        """
        # Try to load existing tracks
        if tracks_path and tracks_path.exists():
            print(f" Loading existing tracks from: {tracks_path}")
            try:
                with open(tracks_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load tracks: {e}. Generating new tracks...")
        
        print("Generating new tracks...")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        tracks = {}
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection and tracking
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    persist=True,
                    verbose=False
                )
                
                # Store track data for this frame
                frame_tracks = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.int().cpu().numpy()
                    
                    for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                        frame_tracks.append({
                            'track_id': int(track_id),
                            'bbox': box.tolist(),  # [x_center, y_center, width, height]
                            'confidence': float(conf),
                            'class': int(cls)
                        })
                
                tracks[frame_idx] = frame_tracks
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"  Processed {frame_idx} frames...")
        
        finally:
            cap.release()
        
        # Save tracks if path provided
        if tracks_path:
            tracks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tracks_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"💾 Tracks saved to: {tracks_path}")
        
        print(f"✅ Generated tracks for {frame_idx} frames")
        return tracks
    
    def _get_center_of_bbox(self, bbox):
        """Get center coordinates of bounding box."""
        x_center, y_center, width, height = bbox
        return int(x_center), int(y_center)
    
    def _get_bbox_width(self, bbox):
        """Get width of bounding box."""
        return int(bbox[2])
    
    def _draw_ellipse(self, frame, bbox, color=(0, 255, 0), label=None):
        """
        Draw ellipse annotation at the bottom of bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box in [x_center, y_center, width, height] format
            color: RGB color tuple
            label: Optional label text
            
        Returns:
            Annotated frame
        """
        x_center, y_center = self._get_center_of_bbox(bbox)
        
        # Get bottom y coordinate
        height = bbox[3]
        y2 = int(y_center + height / 2)
        
        width = self._get_bbox_width(bbox)
        
        # Ensure minimum ellipse dimensions to avoid OpenCV errors
        ellipse_width = max(int(width), 10)  # Minimum width of 10 pixels
        ellipse_height = max(int(0.35 * width), 5)  # Minimum height of 5 pixels
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x_center = max(0, min(x_center, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        
        try:
            # Draw ellipse at bottom of bounding box
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(ellipse_width, ellipse_height),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4
            )
        except cv2.error as e:
            print(f"⚠️  Ellipse drawing error for bbox {bbox}: {e}")
            # Fallback: draw a simple circle
            cv2.circle(frame, (x_center, y2), max(5, ellipse_width // 4), color, 2)
        
        # Draw label if provided
        if label:
            label_x = max(0, min(x_center - 30, frame_width - 100))
            label_y = max(15, y2 - 10)
            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _draw_annotations(self, frame, frame_tracks: list) -> any:
        """
        Draw tracking annotations on a frame using ellipses.
        
        Args:
            frame: Input frame
            frame_tracks: List of track data for this frame
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for track in frame_tracks:
            bbox = track['bbox']  # [x_center, y_center, width, height]
            
            # Create label with track ID and confidence
            label = f"ID:{track['track_id']} ({track['confidence']:.2f})"
            
            # Draw ellipse annotation
            annotated_frame = self._draw_ellipse(
                annotated_frame, 
                bbox, 
                color=(0, 255, 0), 
                label=label
            )
        
        return annotated_frame
    
    
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     tracks_path: Optional[str] = None,
                     use_existing_tracks: bool = True) -> int:
        """
        Process video with object tracking in streaming mode.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            tracks_path: Optional path to save/load track data
            use_existing_tracks: Whether to use existing tracks if available
            
        Returns:
            Number of frames processed
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        tracks_path = Path(tracks_path) if tracks_path else None
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Get video properties
        width, height, total_frames, fps = self._get_video_info(input_path)
        print(f"Video info: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
        
        # Generate or load tracks
        tracks = None
        if use_existing_tracks and tracks_path:
            tracks = self._generate_tracks(input_path, tracks_path)
        
        # Setup video capture and writer
        cap = cv2.VideoCapture(str(input_path))
        writer = self._setup_video_writer(output_path, width, height, fps)
        
        processed_frames = 0
        # Store tracks if we're generating them on-the-fly
        generated_tracks = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get tracking data for current frame
                if tracks and processed_frames in tracks:
                    # Use pre-generated tracks
                    frame_tracks = tracks[processed_frames]
                    annotated_frame = self._draw_annotations(frame, frame_tracks)
                else:
                    # Generate tracks on-the-fly (real-time mode)
                    results = self.model.track(
                        frame,
                        conf=self.confidence_threshold,
                        persist=True,
                        verbose=False
                    )
                    
                    # Store track data if tracks_path is provided and we're not using existing tracks
                    frame_tracks = []
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        classes = results[0].boxes.cls.int().cpu().numpy()
                        
                        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                            frame_tracks.append({
                                'track_id': int(track_id),
                                'bbox': box.tolist(),  # [x_center, y_center, width, height]
                                'confidence': float(conf),
                                'class': int(cls)
                            })
                    
                    # Store tracks for saving later
                    if tracks_path and not use_existing_tracks:
                        generated_tracks[processed_frames] = frame_tracks
                    
                    # Draw annotations
                    if frame_tracks:
                        annotated_frame = self._draw_annotations(frame, frame_tracks)
                    else:
                        annotated_frame = frame.copy()
                
                # Write the annotated frame
                writer.write(annotated_frame)
                processed_frames += 1
                
                # Progress update
                if processed_frames % 100 == 0:
                    progress = (processed_frames / total_frames) * 100
                    print(f"  Progress: {processed_frames}/{total_frames} ({progress:.1f}%)")
        
        finally:
            cap.release()
            writer.release()
        
        # Save generated tracks if we created them and tracks_path is provided
        if tracks_path and not use_existing_tracks and generated_tracks:
            tracks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tracks_path, 'wb') as f:
                pickle.dump(generated_tracks, f)
            print(f"Generated tracks saved to: {tracks_path}")
        
        print(f"Processing complete! {processed_frames} frames processed")
        print(f"Output saved to: {output_path}")
        
        return processed_frames
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold set to: {self.confidence_threshold}")

