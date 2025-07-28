import os
import cv2
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import ultralytics
from ultralytics import YOLO
import sys

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
        
        # Try different codecs for better compatibility
        codecs = ['mp4v', 'XVID', 'MJPG']
        writer = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"✅ Using codec: {codec}")
                break
            writer.release()
        
        if not writer or not writer.isOpened():
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
                print(f" Failed to load tracks: {e}. Generating new tracks...")
        
        print(" Generating new tracks...")
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
                if results and len(results) > 0 and results[0].boxes is not None:
                    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        classes = results[0].boxes.cls.int().cpu().numpy()
                        
                        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                            # Convert goalkeeper (class 1) to player (class 2) immediately
                            converted_cls = int(cls)
                            if converted_cls == 1:  # goalkeeper
                                converted_cls = 2    # convert to player
                            
                            frame_tracks.append({
                                'track_id': int(track_id),
                                'bbox': box.tolist(),  # [x_center, y_center, width, height]
                                'confidence': float(conf),
                                'class': converted_cls  # Store converted class
                            })
                
                tracks[frame_idx] = frame_tracks
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"   Processed {frame_idx} frames...")
        
        finally:
            cap.release()
        
        # Save tracks if path provided
        if tracks_path:
            tracks_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(tracks_path, 'wb') as f:
                    pickle.dump(tracks, f)
                print(f" Tracks saved to: {tracks_path}")
            except Exception as e:
                print(f" Failed to save tracks: {e}")
        
        print(f" Generated tracks for {frame_idx} frames")
        return tracks
    
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
            2: ('player', (0, 255, 0)),      # Green (BGR format) - all players including converted goalkeepers
            3: ('referee', (0, 0, 255))      # Red (BGR format)
        }
        return class_info.get(class_id, ('player', (0, 255, 0)))  # Default to player if unknown
        """Get center coordinates of bounding box."""
        x_center, y_center, width, height = bbox
        return int(x_center), int(y_center)


    def _get_center_of_bbox(self, bbox):
        """Get center coordinates of bounding box."""
        x_center, y_center, width, height = bbox
        return int(x_center), int(y_center)
    
    def _get_bbox_width(self, bbox):
        """Get width of bounding box."""
        return int(bbox[2])
    
    def _draw_ellipse(self, frame, bbox, color=(0, 255, 0), label=None, class_name=None):
        """
        Draw ellipse annotation at the bottom of bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box in [x_center, y_center, width, height] format
            color: BGR color tuple
            label: Optional label text (track ID and confidence)
            class_name: Class name to display
            
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
            
            # Prepare text to display UNDER the ellipse
            text_lines = []
            if class_name:
                text_lines.append(class_name.upper())
            if label:
                text_lines.append(label)
            
            # Draw text UNDER the ellipse if provided
            if text_lines:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                line_height = 20
                
                # Calculate total text area
                max_text_width = 0
                total_text_height = 0
                
                for text in text_lines:
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    max_text_width = max(max_text_width, text_width)
                    total_text_height += text_height + 5  # 5 pixels spacing between lines
                
                # Position text BELOW the ellipse, centered
                text_x = x_center - max_text_width // 2
                text_y_start = y2 + ellipse_height + 15  # Start below the ellipse
                
                # Ensure text is within frame bounds
                text_x = max(5, min(text_x, frame_width - max_text_width - 5))
                text_y_start = max(20, min(text_y_start, frame_height - total_text_height - 5))
                
                # Draw background rectangle for better text visibility
                padding = 3
                cv2.rectangle(frame, 
                            (text_x - padding, text_y_start - 15),
                            (text_x + max_text_width + padding, text_y_start + total_text_height - 10),
                            (0, 0, 0), -1)  # Black background
                
                # Draw each line of text
                current_y = text_y_start
                for text in text_lines:
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    # Center each line
                    line_x = text_x + (max_text_width - text_width) // 2
                    cv2.putText(frame, text, (line_x, current_y), font, font_scale, color, thickness)
                    current_y += text_height + 5
                
        except cv2.error as e:
            print(f" Ellipse drawing error for bbox {bbox}: {e}")
            # Fallback: draw a simple circle with color
            cv2.circle(frame, (x_center, y2), max(5, ellipse_width // 4), color, 2)
            
            # Simple fallback text below the circle
            if class_name:
                cv2.putText(frame, class_name.upper(), (x_center - 20, y2 + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _draw_annotations(self, frame, frame_tracks: list):
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
            
            # Get class-specific color and name
            class_name, color = self.get_class_info(track['class'])
            
            # Create label with track ID and confidence
            label = f"ID:{track['track_id']} ({track['confidence']:.2f})"
            
            # Draw ellipse annotation with class-specific color and labels
            annotated_frame = self._draw_ellipse(
                annotated_frame, 
                bbox, 
                color=color,
                label=label,
                class_name=class_name
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
        
        print(f" Processing video: {input_path}")
        print(f" Output will be saved to: {output_path}")
        
        # Get video properties
        try:
            width, height, total_frames, fps = self._get_video_info(input_path)
            print(f" Video info: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
        except Exception as e:
            print(f" Error getting video info: {e}")
            return 0
        
        # Generate or load tracks
        tracks = None
        if use_existing_tracks and tracks_path:
            try:
                tracks = self._generate_tracks(input_path, tracks_path)
            except Exception as e:
                print(f" Error with tracks: {e}")
                print(" Falling back to real-time tracking...")
        
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
        generated_tracks = {}
        
        print(" Starting video processing...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get tracking data for current frame
                if tracks and processed_frames in tracks:
                    # Use pre-generated tracks
                    frame_tracks = tracks[processed_frames]
                    if frame_tracks:
                        annotated_frame = self._draw_annotations(frame, frame_tracks)
                    else:
                        annotated_frame = frame.copy()
                else:
                    # Generate tracks on-the-fly (real-time mode)
                    try:
                        results = self.model.track(
                            frame,
                            conf=self.confidence_threshold,
                            persist=True,
                            verbose=False
                        )
                        
                        # Extract track data
                        frame_tracks = []
                        if results and len(results) > 0 and results[0].boxes is not None:
                            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                                boxes = results[0].boxes.xywh.cpu().numpy()
                                track_ids = results[0].boxes.id.int().cpu().numpy()
                                confidences = results[0].boxes.conf.cpu().numpy()
                                classes = results[0].boxes.cls.int().cpu().numpy()
                                
                                for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                                    # Convert goalkeeper (class 1) to player (class 2) immediately
                                    converted_cls = int(cls)
                                    if converted_cls == 1:  # goalkeeper
                                        converted_cls = 2    # convert to player
                                    
                                    frame_tracks.append({
                                        'track_id': int(track_id),
                                        'bbox': box.tolist(),  # [x_center, y_center, width, height]
                                        'confidence': float(conf),
                                        'class': converted_cls  # Store converted class
                                    })
                        
                        # Store tracks for saving later
                        if tracks_path and not use_existing_tracks:
                            generated_tracks[processed_frames] = frame_tracks
                        
                        # Draw annotations
                        if frame_tracks:
                            annotated_frame = self._draw_annotations(frame, frame_tracks)
                        else:
                            annotated_frame = frame.copy()
                            
                    except Exception as e:
                        print(f"⚠️ Error processing frame {processed_frames}: {e}")
                        annotated_frame = frame.copy()
                
                # Write the annotated frame
                writer.write(annotated_frame)
                processed_frames += 1
                
                # Progress update
                if processed_frames % 100 == 0:
                    progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                    print(f"   Progress: {processed_frames}/{total_frames} ({progress:.1f}%)")
        
        except Exception as e:
            print(f" Error during processing: {e}")
        finally:
            cap.release()
            writer.release()
        
        # Save generated tracks if we created them and tracks_path is provided
        if tracks_path and not use_existing_tracks and generated_tracks:
            try:
                tracks_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tracks_path, 'wb') as f:
                    pickle.dump(generated_tracks, f)
                print(f" Generated tracks saved to: {tracks_path}")
            except Exception as e:
                print(f" Failed to save generated tracks: {e}")
        
        print(f" Processing complete! {processed_frames} frames processed")
        print(f" Output saved to: {output_path}")
        
        return processed_frames
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f" Confidence threshold set to: {self.confidence_threshold}")

