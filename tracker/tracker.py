import os
import cv2
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import ultralytics
from ultralytics import YOLO
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append('../')

# Define utility functions if my_utils is not available
def get_center_of_bbox(bbox):
    """Get center coordinates of bounding box."""
    x_center, y_center, width, height = bbox
    return int(x_center), int(y_center)

def get_bbox_width(bbox):
    """Get width of bounding box."""
    return int(bbox[2])

class Tracker:
    """
    Memory-efficient video tracker that processes videos in streaming mode.
    Handles object detection, tracking, and annotation rendering without
    loading entire videos into memory.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.05):
        """
        Initialize the Tracker with lower confidence threshold for better ball detection.
        
        Args:
            model_path: Path to the YOLO model weights file
            confidence_threshold: Minimum confidence for detections (lowered to 0.05)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Class-specific confidence thresholds for better detection
        self.class_thresholds = {
            0: 0.25,   # ball - use YOLO's default confidence
            1: 0.1,    # goalkeeper - moderate threshold
            2: 0.1,    # player - moderate threshold  
            3: 0.1     # referee - moderate threshold
        }
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(str(self.model_path))
            print(f"Loaded model: {self.model_path}")
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
        
        # Try different codecs for better compatibility
        codecs = ['mp4v', 'XVID', 'MJPG']
        writer = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Using codec: {codec}")
                break
            writer.release()
        
        if not writer or not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for: {output_path}")
        
        return writer
    
    def _filter_detections_by_class_threshold(self, boxes, track_ids, confidences, classes):
        """
        Filter detections using class-specific confidence thresholds.
        
        Args:
            boxes: Bounding boxes
            track_ids: Track IDs
            confidences: Confidence scores
            classes: Class IDs
            
        Returns:
            Filtered arrays
        """
        keep_indices = []
        
        for i, (conf, cls) in enumerate(zip(confidences, classes)):
            cls = int(cls)
            threshold = self.class_thresholds.get(cls, self.confidence_threshold)
            
            if conf >= threshold:
                keep_indices.append(i)
        
        if keep_indices:
            return (boxes[keep_indices], 
                   track_ids[keep_indices], 
                   confidences[keep_indices], 
                   classes[keep_indices])
        else:
            return None, None, None, None
    
    def _get_ball_detections_from_video(self, video_path: Path):
        """
        Run YOLO predict directly on video file to get ONLY ball detections.
        This replicates the exact behavior of model.predict('video.mp4', save=True)
        but filters to only ball detections.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary mapping frame_idx to list of ball detections
        """
        print("Running YOLO predict on entire video for ball detection...")
        
        # Run predict on the entire video with stream=True to get frame-by-frame results
        results_generator = self.model.predict(
            str(video_path),
            stream=True,     # Process frame by frame
            conf=0.25,       # Default confidence
            save=False,      # Don't save, we'll handle that
            verbose=False,
            classes=[0]      # Only detect balls (class 0)
        )
        
        ball_detections = {}
        frame_idx = 0
        ball_count = 0
        
        for result in results_generator:
            frame_balls = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywh.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.int().cpu().numpy()
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    if int(cls) == 0:  # Ball class
                        ball_count += 1
                        frame_balls.append({
                            'track_id': f"ball_{frame_idx}_{i}",  # Unique ball ID per frame
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class': 0,
                            'original_class': 0
                        })
            
            ball_detections[frame_idx] = frame_balls
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                recent_balls = sum(1 for i in range(max(0, frame_idx-100), frame_idx) 
                                 for ball in ball_detections.get(i, []))
                print(f"   Processed {frame_idx} frames... (balls in last 100 frames: {recent_balls})")
        
        total_ball_detections = sum(len(balls) for balls in ball_detections.values())
        frames_with_balls = sum(1 for balls in ball_detections.values() if len(balls) > 0)
        ball_frame_percentage = (frames_with_balls / frame_idx) * 100 if frame_idx > 0 else 0
        
        print(f"Ball detection results from video predict:")
        print(f"  Total frames: {frame_idx}")
        print(f"  Total ball detections: {total_ball_detections}")
        print(f"  Frames with balls: {frames_with_balls} ({ball_frame_percentage:.1f}%)")
        
        return ball_detections
    
    def _get_tracked_detections_from_video(self, video_path: Path):
        """
        Run YOLO track on video to get tracked detections for players, goalkeepers, referees.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary mapping frame_idx to list of tracked detections
        """
        print("Running YOLO track on entire video for player/referee tracking...")
        
        # Run track on the entire video for non-ball objects
        results_generator = self.model.track(
            str(video_path),
            stream=True,     # Process frame by frame
            conf=0.25,       # Default confidence
            persist=True,    # Maintain tracks across frames
            verbose=False,
            classes=[1, 2, 3]  # Only track players, goalkeepers, referees
        )
        
        tracked_detections = {}
        frame_idx = 0
        
        for result in results_generator:
            frame_tracks = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.int().cpu().numpy()
                    
                    for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                        original_cls = int(cls)
                        # Convert goalkeeper (class 1) to player (class 2)
                        converted_cls = original_cls
                        if converted_cls == 1:  # goalkeeper
                            converted_cls = 2    # convert to player
                        
                        frame_tracks.append({
                            'track_id': int(track_id),
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class': converted_cls,
                            'original_class': original_cls
                        })
            
            tracked_detections[frame_idx] = frame_tracks
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                recent_tracks = sum(len(tracks) for i in range(max(0, frame_idx-100), frame_idx) 
                                  for tracks in [tracked_detections.get(i, [])])
                print(f"   Processed {frame_idx} frames... (tracked objects in last 100 frames: {recent_tracks})")
        
        total_tracked = sum(len(tracks) for tracks in tracked_detections.values())
        print(f"Tracking results:")
        print(f"  Total frames: {frame_idx}")
        print(f"  Total tracked detections: {total_tracked}")
        
        return tracked_detections

    def _generate_tracks(self, video_path: Path, tracks_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate tracks using hybrid approach:
        - YOLO predict for balls (like your working method)
        - YOLO track for players/referees (for consistent tracking)
        """
        # Try to load existing tracks
        if tracks_path and tracks_path.exists():
            print(f"Loading existing tracks from: {tracks_path}")
            try:
                with open(tracks_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load tracks: {e}. Generating new tracks...")
        
        print("Generating hybrid tracks: predict for balls, track for players...")
        
        # Get ball detections using video predict (your working method)
        ball_detections = self._get_ball_detections_from_video(video_path)
        
        # Get tracked detections for players/referees
        tracked_detections = self._get_tracked_detections_from_video(video_path)
        
        # Combine ball detections and tracked detections
        combined_tracks = {}
        max_frames = max(len(ball_detections), len(tracked_detections))
        
        for frame_idx in range(max_frames):
            frame_tracks = []
            
            # Add ball detections for this frame
            if frame_idx in ball_detections:
                frame_tracks.extend(ball_detections[frame_idx])
            
            # Add tracked detections for this frame
            if frame_idx in tracked_detections:
                frame_tracks.extend(tracked_detections[frame_idx])
            
            combined_tracks[frame_idx] = frame_tracks
        
        # Print combined statistics
        total_balls = sum(len([t for t in tracks if t['original_class'] == 0]) 
                         for tracks in combined_tracks.values())
        total_players = sum(len([t for t in tracks if t['original_class'] in [1, 2, 3]]) 
                           for tracks in combined_tracks.values())
        frames_with_balls = sum(1 for tracks in combined_tracks.values() 
                               if any(t['original_class'] == 0 for t in tracks))
        
        print(f"Combined tracking results:")
        print(f"  Total frames: {max_frames}")
        print(f"  Ball detections: {total_balls}")
        print(f"  Player/referee detections: {total_players}")
        print(f"  Frames with balls: {frames_with_balls} ({frames_with_balls/max_frames*100:.1f}%)")
        
        # Save tracks if path provided
        if tracks_path:
            tracks_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(tracks_path, 'wb') as f:
                    pickle.dump(combined_tracks, f)
                print(f"Combined tracks saved to: {tracks_path}")
            except Exception as e:
                print(f"Failed to save tracks: {e}")
        
        return combined_tracks
    
    def get_class_info(self, class_id):
        """
        Get class name and color from class ID (after goalkeeper->player conversion).
        
        Args:
            class_id: Class ID after conversion
            
        Returns:
            Tuple of (class_name, color_bgr)
        """
        class_info = {
            0: ('ball', (0, 255, 255)),      # Yellow (BGR format)
            1: ('goalkeeper', (255, 0, 255)), # Magenta (BGR format) - will be converted to player
            2: ('player', (0, 255, 0)),      # Green (BGR format) - all players including converted goalkeepers
            3: ('referee', (0, 0, 255))      # Red (BGR format)
        }
        return class_info.get(class_id, (f'class_{class_id}', (128, 128, 128)))  # Gray for unknown classes

    def _get_center_of_bbox(self, bbox):
        """Get center coordinates of bounding box."""
        x_center, y_center, width, height = bbox
        return int(x_center), int(y_center)
    
    def _get_bbox_width(self, bbox):
        """Get width of bounding box."""
        return int(bbox[2])
    
    def _draw_triangle_above_ball(self, frame, bbox, color=(0, 255, 255)):
        """
        Draw a triangle above the ball, pointing downward toward it.
        
        Args:
            frame: Input frame
            bbox: Bounding box in [x_center, y_center, width, height] format
            color: BGR color tuple
            
        Returns:
            Annotated frame
        """
        x_center, y_center = self._get_center_of_bbox(bbox)
        
        # Get top y coordinate of the ball
        height = bbox[3]
        y_top = int(y_center - height / 2)
        
        width = self._get_bbox_width(bbox)
        
        # Triangle dimensions
        triangle_size = max(int(width * 0.3), 20)  # Minimum size of 20 pixels
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x_center = max(0, min(x_center, frame_width - 1))
        
        # Triangle points (pointing downward toward the ball)
        triangle_bottom = (x_center, max(0, y_top - 5))  # Bottom vertex near the ball
        triangle_top_left = (max(0, x_center - triangle_size // 2), max(0, y_top - triangle_size - 5))  # Top-left of base
        triangle_top_right = (min(frame_width - 1, x_center + triangle_size // 2), max(0, y_top - triangle_size - 5))  # Top-right of base
        
        # Draw filled triangle
        triangle_points = np.array([triangle_bottom, triangle_top_left, triangle_top_right], np.int32)
        cv2.fillPoly(frame, [triangle_points], color)
        
        # Draw triangle outline for better visibility
        cv2.polylines(frame, [triangle_points], True, (0, 0, 0), 1)  # Black outline
        
        return frame
    
    def _draw_ellipse(self, frame, bbox, color=(0, 255, 0), label=None):
        """
        Draw ellipse annotation at the bottom of bounding box (only for non-ball objects).
        
        Args:
            frame: Input frame
            bbox: Bounding box in [x_center, y_center, width, height] format
            color: BGR color tuple
            label: Optional label text (track ID only)
            
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
            # Draw ellipse at bottom of bounding box with class-specific color
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
            
            # Draw only the ID text UNDER the ellipse if provided
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Calculate text dimensions
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Position text BELOW the ellipse, centered
                text_x = x_center - text_width // 2
                text_y = y2 + ellipse_height + 20  # Start below the ellipse
                
                # Ensure text is within frame bounds
                text_x = max(5, min(text_x, frame_width - text_width - 5))
                text_y = max(20, min(text_y, frame_height - 5))
                
                # Draw background rectangle for better text visibility
                padding = 3
                cv2.rectangle(frame, 
                            (text_x - padding, text_y - text_height - padding),
                            (text_x + text_width + padding, text_y + padding),
                            color, -1)  # Same color as ellipse
                
                # Draw the ID text in white for contrast
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
        except cv2.error as e:
            print(f"Ellipse drawing error for bbox {bbox}: {e}")
            # Fallback: draw a simple circle with color
            cv2.circle(frame, (x_center, y2), max(5, ellipse_width // 4), color, 2)
            
            # Simple fallback text below the circle
            if label:
                cv2.putText(frame, label, (x_center - 20, y2 + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _draw_annotations(self, frame, frame_tracks: list):
        """
        Draw tracking annotations on a frame using triangles for balls and ellipses for other objects.
        
        Args:
            frame: Input frame
            frame_tracks: List of track data for this frame
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for track in frame_tracks:
            bbox = track['bbox']  # [x_center, y_center, width, height]
            class_id = track['class']
            
            # Get class-specific color and name
            class_name, color = self.get_class_info(class_id)
            
            if class_id == 0:  # Ball
                # Draw triangle above the ball (no ID or class name)
                annotated_frame = self._draw_triangle_above_ball(
                    annotated_frame, 
                    bbox, 
                    color=color
                )
            else:  # Other objects (players, referees)
                # Create label with only track ID (no class name, no confidence)
                label = f"ID:{track['track_id']}"
                
                # Draw ellipse annotation with only ID
                annotated_frame = self._draw_ellipse(
                    annotated_frame, 
                    bbox, 
                    color=color,
                    label=label
                )
        
        return annotated_frame
    
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     tracks_path: Optional[str] = None,
                     use_existing_tracks: bool = True) -> int:
        """
        Process video using YOLO predict on entire video (like your working method).
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        tracks_path = Path(tracks_path) if tracks_path else None
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Using hybrid approach: video predict for balls, video track for players")
        
        # Get video properties
        try:
            width, height, total_frames, fps = self._get_video_info(input_path)
            print(f"Video info: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
        except Exception as e:
            print(f"Error getting video info: {e}")
            return 0
        
        # Generate or load tracks using hybrid approach
        tracks = None
        if use_existing_tracks and tracks_path:
            try:
                tracks = self._generate_tracks(input_path, tracks_path)
            except Exception as e:
                print(f"Error with tracks: {e}")
                print("Falling back to real-time tracking...")
        
        # Setup video capture and writer
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        try:
            writer = self._setup_video_writer(output_path, width, height, fps)
        except Exception as e:
            cap.release()
            raise RuntimeError(f"Failed to setup video writer: {e}")
        
        processed_frames = 0
        ball_frames_count = 0
        
        print("Starting video annotation...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get tracking data for current frame
                if tracks and processed_frames in tracks:
                    # Use pre-generated hybrid tracks
                    frame_tracks = tracks[processed_frames]
                    if frame_tracks:
                        annotated_frame = self._draw_annotations(frame, frame_tracks)
                        # Count frames with balls
                        if any(track['original_class'] == 0 for track in frame_tracks):
                            ball_frames_count += 1
                    else:
                        annotated_frame = frame.copy()
                else:
                    # Fallback: no pre-generated tracks available
                    annotated_frame = frame.copy()
                
                # Write the annotated frame
                writer.write(annotated_frame)
                processed_frames += 1
                
                # Progress update with ball statistics
                if processed_frames % 100 == 0:
                    progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                    ball_percentage = (ball_frames_count / processed_frames) * 100
                    print(f"   Progress: {processed_frames}/{total_frames} ({progress:.1f}%) - Ball presence: {ball_percentage:.1f}%")
        
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            cap.release()
            writer.release()
        
        # Final statistics
        final_ball_percentage = (ball_frames_count / processed_frames) * 100 if processed_frames > 0 else 0
        print(f"Final ball detection statistics:")
        print(f"  Frames with ball triangles: {ball_frames_count}/{processed_frames} ({final_ball_percentage:.1f}%)")
        
        print(f"Processing complete! {processed_frames} frames processed")
        print(f"Output saved to: {output_path}")
        
        return processed_frames
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold set to: {self.confidence_threshold}")