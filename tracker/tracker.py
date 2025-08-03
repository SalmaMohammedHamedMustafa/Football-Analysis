import os
import cv2
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import ultralytics
from ultralytics import YOLO
import sys
from sklearn.cluster import KMeans
sys.path.append('../')
from team_assigner import TeamAssigner
from my_utils import get_bbox_width, get_center_of_bbox, get_bbox_height
from ball_control_tracker import BallControlTracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


class Tracker:
    """
    Memory-efficient video tracker that processes videos in streaming mode.
    Handles object detection, tracking, and annotation rendering without
    loading entire videos into memory.
    Enhanced with pandas-based linear interpolation for missing ball positions.
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
        self.team_assigner = TeamAssigner()
        self.team_colors_assigned = False
        self.camera_movement_estimator = None
        self.camera_movement_per_frame = None
        self.view_transformer = ViewTransformer()
        
        # Class-specific confidence thresholds for better detection
        self.class_thresholds = {
            0: 0.25,   # ball - use YOLO's default confidence
            1: 0.1,    # goalkeeper - moderate threshold
            2: 0.1,    # player - moderate threshold  
            3: 0.1     # referee - moderate threshold
        }
        self.ball_control_tracker = BallControlTracker(
            possession_distance_threshold=100.0,  # Adjust based on your video resolution
            min_frames_for_possession=3,          # Frames needed to confirm possession change
            possession_smoothing_window=5         # Smoothing window size
            )
        
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


    def _setup_camera_movement_estimation(self, video_path: Path, camera_movement_stub_path: Optional[Path] = None, use_camera_movement: bool = True):
        """
        Setup camera movement estimation for the video.
        
        Args:
            video_path: Path to input video
            camera_movement_stub_path: Path to save/load camera movement data
            use_camera_movement: Whether to enable camera movement compensation
        """
        if not use_camera_movement:
            print("Camera movement compensation disabled")
            return
        
        print("Setting up camera movement estimation...")
        
        # Get first frame for initialization
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video for camera movement setup: {video_path}")
        
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Cannot read first frame for camera movement setup")
        
        # Initialize camera movement estimator
        self.camera_movement_estimator = CameraMovementEstimator(first_frame)
        
        # Calculate camera movement for entire video
        self.camera_movement_per_frame = self.camera_movement_estimator.get_camera_movement_streaming(
            video_path, 
            read_from_stub=True if camera_movement_stub_path else False,
            stub_path=camera_movement_stub_path
        )
        
        print(f"Camera movement estimation complete: {len(self.camera_movement_per_frame)} frames")
    
    def _adjust_tracks_for_camera_movement(self, tracks: list, frame_idx: int) -> list:
        """
        Adjust track positions for camera movement and add transformed positions.
        
        Args:
            tracks: List of tracks for current frame
            frame_idx: Current frame index
            
        Returns:
            Adjusted tracks with transformed positions
        """
        if (self.camera_movement_estimator is None or 
            self.camera_movement_per_frame is None or 
            frame_idx >= len(self.camera_movement_per_frame)):
            adjusted_tracks = tracks
        else:
            camera_movement = self.camera_movement_per_frame[frame_idx]
            adjusted_tracks = []
            
            for track in tracks:
                adjusted_track = track.copy()
                
                # Adjust bounding box for camera movement
                adjusted_bbox = self.camera_movement_estimator.adjust_bbox_for_camera_movement(
                    track['bbox'], camera_movement
                )
                adjusted_track['bbox'] = adjusted_bbox
                
                # Store original and adjusted positions for analysis
                original_center = get_center_of_bbox(track['bbox'])
                adjusted_center = get_center_of_bbox(adjusted_bbox)
                
                adjusted_track['position_original'] = original_center
                adjusted_track['position_adjusted'] = adjusted_center
                adjusted_track['camera_movement'] = camera_movement
                
                adjusted_tracks.append(adjusted_track)
        
        # Add transformed positions for all tracks
        for track in adjusted_tracks:
            # Use adjusted position if available, otherwise use original position
            if 'position_adjusted' in track:
                position = np.array(track['position_adjusted'])
            else:
                position = np.array(get_center_of_bbox(track['bbox']))
            
            # Transform the position to real-world coordinates
            position_transformed = self.view_transformer.transform_point(position)
            if position_transformed is not None:
                position_transformed = position_transformed.squeeze().tolist()
            
            track['position_transformed'] = position_transformed
        
        return adjusted_tracks
    
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
            save=False,      
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

    def _interpolate_ball_positions(self, ball_detections: Dict[int, list], total_frames: int) -> Dict[int, list]:
        """
        Apply pandas linear interpolation to fill missing ball positions (center only).
        Returns the same structure as the input, with interpolated values inserted where needed.
        """
        # Collect center positions as a simple per-frame list
        centers = []
        widths = []
        heights = []

        for frame_idx in range(total_frames):
            if frame_idx in ball_detections and len(ball_detections[frame_idx]) > 0:
                bbox = ball_detections[frame_idx][0]['bbox']  # [x_center, y_center, width, height]
                x, y = get_center_of_bbox(bbox)
                centers.append({'x': x, 'y': y})
                widths.append(get_bbox_width(bbox))
                heights.append(get_bbox_height(bbox))
            else:
                centers.append({'x': np.nan, 'y': np.nan})
                widths.append(np.nan)
                heights.append(np.nan)

        # Interpolate x/y center columns independently in a DataFrame
        df = pd.DataFrame(centers)
        df['width'] = widths
        df['height'] = heights

        df_interp = df.interpolate(method='linear', limit_direction='both')
        # Fill leading/trailing NaNs if any (optional, but matches .bfill() in your example)
        df_interp = df_interp.bfill().ffill()

        # Build output dict, inserting interpolated center for missing frames
        updated_ball_detections = ball_detections.copy()
        for frame_idx in range(total_frames):
            # Only fill where detection was missing
            if frame_idx not in ball_detections or len(ball_detections[frame_idx]) == 0:
                interp_x = float(df_interp.loc[frame_idx, 'x'])
                interp_y = float(df_interp.loc[frame_idx, 'y'])
                interp_w = float(df_interp.loc[frame_idx, 'width']) if not np.isnan(df_interp.loc[frame_idx, 'width']) else 30
                interp_h = float(df_interp.loc[frame_idx, 'height']) if not np.isnan(df_interp.loc[frame_idx, 'height']) else 30
                ball_det = {
                    'track_id': f"ball_interpolated_{frame_idx}",
                    'bbox': [interp_x, interp_y, interp_w, interp_h],
                    'confidence': 0.5,
                    'class': 0,
                    'original_class': 0,
                    'interpolated': True
                }
                updated_ball_detections[frame_idx] = [ball_det]
        return updated_ball_detections

    
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

    def _generate_tracks_with_camera_adjustment(self, video_path: Path, tracks_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate tracks with camera movement adjustment and view transformation.
        This is your existing _generate_tracks method with camera movement integration and view transformation.
        """
        # Try to load existing tracks
        if tracks_path and tracks_path.exists():
            print(f"Loading existing tracks from: {tracks_path}")
            try:
                with open(tracks_path, 'rb') as f:
                    tracks = pickle.load(f)
                    
                # Check if tracks already have camera movement adjustment and view transformation
                sample_tracks = next(iter(tracks.values())) if tracks else []
                has_camera_adjustment = any('position_adjusted' in track for track in sample_tracks)
                has_view_transformation = any('position_transformed' in track for track in sample_tracks)
                
                if has_camera_adjustment and has_view_transformation:
                    print("Tracks already contain camera movement adjustments and view transformations")
                    return tracks
                elif has_camera_adjustment and not has_view_transformation:
                    print("Tracks have camera movement adjustments but need view transformation")
                    # Add view transformations to existing tracks
                    for frame_idx, frame_tracks in tracks.items():
                        for track in frame_tracks:
                            if 'position_adjusted' in track:
                                position = np.array(track['position_adjusted'])
                            else:
                                position = np.array(get_center_of_bbox(track['bbox']))
                            
                            position_transformed = self.view_transformer.transform_point(position)
                            if position_transformed is not None:
                                position_transformed = position_transformed.squeeze().tolist()
                            
                            track['position_transformed'] = position_transformed
                    
                    # Save updated tracks
                    if tracks_path:
                        with open(tracks_path, 'wb') as f:
                            pickle.dump(tracks, f)
                        print("Updated tracks with view transformations saved")
                    
                    return tracks
                else:
                    print("Tracks loaded but need camera movement adjustment and view transformation")
                    
            except Exception as e:
                print(f"Failed to load tracks: {e}. Generating new tracks...")
        
        # Generate tracks using your existing method
        print("Generating tracks with camera movement adjustment and view transformation...")
        
        # Get video info
        width, height, total_frames, fps = self._get_video_info(video_path)
        
        # Get ball detections using video predict
        ball_detections = self._get_ball_detections_from_video(video_path)
        
        # Apply interpolation to ball positions
        interpolated_ball_detections = self._interpolate_ball_positions(ball_detections, total_frames)
        
        # Get tracked detections for players/referees
        tracked_detections = self._get_tracked_detections_from_video(video_path)
        
        # Combine detections
        combined_tracks = {}
        max_frames = max(len(interpolated_ball_detections), len(tracked_detections))
        
        for frame_idx in range(max_frames):
            frame_tracks = []
            
            # Add ball detections
            if frame_idx in interpolated_ball_detections:
                frame_tracks.extend(interpolated_ball_detections[frame_idx])
            
            # Add tracked detections
            if frame_idx in tracked_detections:
                frame_tracks.extend(tracked_detections[frame_idx])
            
            # Apply camera movement adjustment and view transformation
            frame_tracks = self._adjust_tracks_for_camera_movement(frame_tracks, frame_idx)
            
            combined_tracks[frame_idx] = frame_tracks
        
        # Save tracks if path provided
        if tracks_path:
            tracks_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(tracks_path, 'wb') as f:
                    pickle.dump(combined_tracks, f)
                print(f"Tracks with camera adjustment and view transformation saved to: {tracks_path}")
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
        x_center, y_center = get_center_of_bbox(bbox)
        
        # Get top y coordinate of the ball
        height = bbox[3]
        y_top = int(y_center - height / 2)
        
        width = get_bbox_width(bbox)
        
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
        def get_contrasting_text_color(bg_color):
            """
            Calculate contrasting text color based on background color brightness.
            Args:
                bg_color: BGR color tuple
            Returns:
                BGR color tuple for text (either black or white)
            """
            # Convert BGR to grayscale using standard luminance formula
            b, g, r = bg_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b

            # If background is bright, use black text; if dark, use white text
            return (0, 0, 0) if luminance > 127 else (255, 255, 255)

        x_center, y_center = get_center_of_bbox(bbox)
        # Get bottom y coordinate
        height = bbox[3]
        y2 = int(y_center + height / 2)
        width = get_bbox_width(bbox)

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

                # Get contrasting text color based on ellipse color
                text_color = get_contrasting_text_color(color)

                # Draw background rectangle for better text visibility
                padding = 3
                cv2.rectangle(frame,
                              (text_x - padding, text_y - text_height - padding),
                              (text_x + text_width + padding, text_y + padding),
                              color, -1)  # Same color as ellipse

                # Draw the ID text with contrasting color
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness)

        except cv2.error as e:
            print(f"Ellipse drawing error for bbox {bbox}: {e}")
            # Fallback: draw a simple circle with color
            cv2.circle(frame, (x_center, y2), max(5, ellipse_width // 4), color, 2)

            # Simple fallback text below the circle with contrasting color
            if label:
                text_color = get_contrasting_text_color(color)
                cv2.putText(frame, label, (x_center - 20, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return frame

    def _draw_annotations(self, frame, frame_tracks: list, frame_idx: int = 0, show_camera_movement: bool = True):
        """
        Draw tracking annotations on a frame using triangles for balls and ellipses for other objects.
        Enhanced with team-based coloring for players, visual indication for interpolated balls,
        and ball control/possession tracking.
        
        Args:
            frame: Input frame
            frame_tracks: List of track data for this frame
            frame_idx: Current frame index for possession tracking
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        if (show_camera_movement and 
            self.camera_movement_per_frame is not None and 
            self.camera_movement_estimator is not None):
            annotated_frame = self.camera_movement_estimator.draw_camera_movement(
                annotated_frame, self.camera_movement_per_frame, frame_idx
            )
        
        # Try to assign team colors if not already done
        self._assign_teams_for_frame(annotated_frame, frame_tracks)
        
        # Extract ball and player information for possession analysis
        ball_bbox = None
        player_tracks = []
        
        # Separate ball and player tracks for possession analysis
        for track in frame_tracks:
            if track['class'] == 0:  # Ball
                ball_bbox = track['bbox']
            elif track['class'] == 2:  # Players (including converted goalkeepers)
                player_tracks.append(track)
        
        # Update ball control analysis
        current_possession = None
        if self.team_colors_assigned:  # Only track possession after teams are assigned
            current_possession = self.ball_control_tracker.update_ball_control(
                frame_idx=frame_idx,
                ball_bbox=ball_bbox,
                player_tracks=player_tracks,
                team_assignments=self.team_assigner.player_team_dict
            )
        
        # Draw possession indicator if we have possession data
        if current_possession is not None:
            team_colors = self.get_team_colors()
            annotated_frame = self.ball_control_tracker.draw_possession_indicator(
                annotated_frame, current_possession, team_colors
            )
        
        # Draw annotations for all tracked objects
        for track in frame_tracks:
            bbox = track['bbox']  # [x_center, y_center, width, height]
            class_id = track['class']
            track_id = track['track_id']
            
            if class_id == 0:  # Ball
                # Get class-specific color and name for ball
                class_name, color = self.get_class_info(class_id)
                
                # Use different color for interpolated balls to distinguish them visually
                if track.get('interpolated', False):
                    color = (0, 200, 200)  # Slightly different yellow/cyan for interpolated balls
                
                # Draw triangle above the ball (no ID or class name)
                annotated_frame = self._draw_triangle_above_ball(
                    annotated_frame, 
                    bbox, 
                    color=color
                )
            else:  # Other objects (players, referees)
                if class_id == 2:  # Players (including converted goalkeepers)
                    # Get team assignment and use team color
                    try:
                        team_id = self.team_assigner.get_player_team(annotated_frame, bbox, track_id)
                        color = self.team_assigner.get_team_color_bgr(team_id)
                        label = f"ID:{track_id} T{team_id}"
                    except Exception as e:
                        # Fallback to default player color
                        class_name, color = self.get_class_info(class_id)
                        label = f"ID:{track_id}"
                        print(f"Error getting team for player {track_id}: {e}")
                else:
                    # Referees and other objects use default colors
                    class_name, color = self.get_class_info(class_id)
                    label = f"ID:{track_id}"
                
                # Draw ellipse annotation
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
            camera_movement_stub_path: Optional[str] = None,
            use_existing_tracks: bool = True,
            use_camera_movement: bool = True,
            show_camera_movement: bool = True) -> int:
        """
        Process video using YOLO predict on entire video with ball position interpolation.
        Enhanced with team assignment functionality, pandas-based interpolation, ball control tracking,
        camera movement compensation, and view transformation.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        tracks_path = Path(tracks_path) if tracks_path else None
        camera_movement_stub_path = Path(camera_movement_stub_path) if camera_movement_stub_path else None
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Using hybrid approach with ball interpolation, possession tracking, and view transformation: video predict for balls + pandas interpolation, video track for players")
        print(f"Camera movement compensation: {'Enabled' if use_camera_movement else 'Disabled'}")
        print(f"View transformation: Enabled (court coordinates)")
        
        # Get video properties
        try:
            width, height, total_frames, fps = self._get_video_info(input_path)
            print(f"Video info: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
        except Exception as e:
            print(f"Error getting video info: {e}")
            return 0
        
        # Setup camera movement estimation BEFORE generating tracks
        if use_camera_movement:
            print("Setting up camera movement estimation...")
            try:
                self._setup_camera_movement_estimation(
                    input_path, 
                    camera_movement_stub_path,
                    use_camera_movement
                )
                print(f"Camera movement estimation complete: {len(self.camera_movement_per_frame)} frames")
            except Exception as e:
                print(f"Error setting up camera movement: {e}")
                print("Continuing without camera movement compensation...")
                use_camera_movement = False
        
        # Reset ball control tracker for new video
        self.ball_control_tracker.reset()
        
        # Auto-adjust possession distance threshold based on video resolution
        video_resolution_factor = min(width, height) / 720  # Base on 720p
        adjusted_distance_threshold = 100.0 * video_resolution_factor
        self.ball_control_tracker.possession_distance_threshold = adjusted_distance_threshold
        print(f"Ball possession distance threshold auto-adjusted to: {adjusted_distance_threshold:.1f} pixels")
        
        # Generate or load tracks using hybrid approach with interpolation, camera movement, and view transformation
        tracks = None
        if use_existing_tracks and tracks_path:
            try:
                if use_camera_movement:
                    tracks = self._generate_tracks_with_camera_adjustment(input_path, tracks_path)
                else:
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
        interpolated_ball_frames_count = 0
        team_assignment_frame = -1
        possession_start_frame = -1
        transformed_positions_count = 0
        
        print("Starting video annotation with team assignment, interpolated ball positions, possession tracking, and view transformation...")
        if use_camera_movement:
            print("Camera movement compensation is active")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get tracking data for current frame
                if tracks and processed_frames in tracks:
                    # Use pre-generated hybrid tracks with interpolation, camera adjustment, and view transformation
                    frame_tracks = tracks[processed_frames]
                    if frame_tracks:
                        # Pass frame index for possession tracking and show camera movement if enabled
                        annotated_frame = self._draw_annotations(
                            frame, frame_tracks, processed_frames, show_camera_movement
                        )
                        
                        # Count frames with balls and interpolated balls
                        if any(track['original_class'] == 0 for track in frame_tracks):
                            ball_frames_count += 1
                            if any(track.get('interpolated', False) for track in frame_tracks if track['original_class'] == 0):
                                interpolated_ball_frames_count += 1
                        
                        # Count transformed positions
                        transformed_positions_count += sum(1 for track in frame_tracks if track.get('position_transformed') is not None)
                        
                        # Record when team assignment happened
                        if self.team_colors_assigned and team_assignment_frame == -1:
                            team_assignment_frame = processed_frames
                        
                        # Record when possession tracking started
                        current_possession = self.ball_control_tracker.get_current_possession()
                        if current_possession is not None and possession_start_frame == -1:
                            possession_start_frame = processed_frames
                    else:
                        annotated_frame = frame.copy()
                        
                        # Still show camera movement even if no tracks
                        if (show_camera_movement and use_camera_movement and 
                            self.camera_movement_per_frame is not None and 
                            self.camera_movement_estimator is not None):
                            annotated_frame = self.camera_movement_estimator.draw_camera_movement(
                                annotated_frame, self.camera_movement_per_frame, processed_frames
                            )
                else:
                    # Fallback: no pre-generated tracks available
                    annotated_frame = frame.copy()
                    
                    # Still show camera movement even without tracks
                    if (show_camera_movement and use_camera_movement and 
                        self.camera_movement_per_frame is not None and 
                        self.camera_movement_estimator is not None):
                        annotated_frame = self.camera_movement_estimator.draw_camera_movement(
                            annotated_frame, self.camera_movement_per_frame, processed_frames
                        )
                
                # Write the annotated frame
                writer.write(annotated_frame)
                processed_frames += 1
                
                # Progress update with ball, interpolation, team, possession, camera movement, and view transformation statistics
                if processed_frames % 100 == 0:
                    progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                    ball_percentage = (ball_frames_count / processed_frames) * 100
                    interpolated_percentage = (interpolated_ball_frames_count / processed_frames) * 100
                    team_status = "assigned" if self.team_colors_assigned else "pending"
                    
                    # Get current possession info for progress updates
                    current_possession = self.ball_control_tracker.get_current_possession()
                    possession_status = f"Team {current_possession}" if current_possession else "none"
                    
                    # Camera movement status for progress
                    camera_status = "enabled" if use_camera_movement else "disabled"
                    
                    # View transformation status
                    transform_percentage = (transformed_positions_count / (processed_frames * 10)) * 100 if processed_frames > 0 else 0  # Rough estimate
                    
                    print(f"   Progress: {processed_frames}/{total_frames} ({progress:.1f}%) - Ball: {ball_percentage:.1f}% (interp: {interpolated_percentage:.1f}%) - Teams: {team_status} - Possession: {possession_status} - Camera: {camera_status} - Transforms: {transform_percentage:.1f}%")
        
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            cap.release()
            writer.release()
        
        # Final statistics with interpolation, possession, camera movement, and view transformation details
        final_ball_percentage = (ball_frames_count / processed_frames) * 100 if processed_frames > 0 else 0
        final_interpolated_percentage = (interpolated_ball_frames_count / processed_frames) * 100 if processed_frames > 0 else 0
        
        print(f"\n" + "="*60)
        print(f"FINAL PROCESSING STATISTICS")
        print(f"="*60)
        
        # Ball detection statistics
        print(f"Ball Detection:")
        print(f"  Frames with ball triangles: {ball_frames_count}/{processed_frames} ({final_ball_percentage:.1f}%)")
        print(f"  Frames with interpolated balls: {interpolated_ball_frames_count}/{processed_frames} ({final_interpolated_percentage:.1f}%)")
        
        # Team assignment statistics
        if self.team_colors_assigned:
            print(f"\nTeam Assignment:")
            print(f"  Team colors assigned at frame: {team_assignment_frame}")
            print(f"  Team 1 color (BGR): {self.team_assigner.get_team_color_bgr(1)}")
            print(f"  Team 2 color (BGR): {self.team_assigner.get_team_color_bgr(2)}")
            print(f"  Players assigned to teams: {len(self.team_assigner.player_team_dict)}")
            
            # Display team assignments
            team_1_players = [pid for pid, tid in self.team_assigner.player_team_dict.items() if tid == 1]
            team_2_players = [pid for pid, tid in self.team_assigner.player_team_dict.items() if tid == 2]
            print(f"  Team 1 players: {sorted(team_1_players)}")
            print(f"  Team 2 players: {sorted(team_2_players)}")
        else:
            print(f"\nTeam Assignment:")
            print(f"  Team colors: Not assigned (insufficient players)")
        
        # Ball possession statistics
        possession_stats = self.ball_control_tracker.get_possession_stats(processed_frames)
        print(f"\nBall Possession Analysis:")
        print(f"  Possession tracking started at frame: {possession_start_frame if possession_start_frame != -1 else 'N/A'}")
        print(f"  Total frames with possession data: {possession_stats['frames_with_possession']}/{possession_stats['total_frames']} ({possession_stats['possession_percentage']:.1f}%)")
        
        if possession_stats['frames_with_possession'] > 0:
            print(f"  Team 1 possession: {possession_stats['team_1_frames']} frames ({possession_stats['team_1_percentage']:.1f}%)")
            print(f"  Team 2 possession: {possession_stats['team_2_frames']} frames ({possession_stats['team_2_percentage']:.1f}%)")
            print(f"  Possession changes: {possession_stats['possession_changes']}")
            print(f"  Average possession duration: {possession_stats['average_possession_duration']:.1f} frames ({possession_stats['average_possession_duration']/fps:.1f} seconds)")
            
            # Calculate possession ratio
            if possession_stats['team_1_percentage'] > 0 and possession_stats['team_2_percentage'] > 0:
                ratio = possession_stats['team_1_percentage'] / possession_stats['team_2_percentage']
                print(f"  Possession ratio (Team 1:Team 2): {ratio:.2f}:1")
            
            # Possession quality metrics
            if possession_stats['possession_changes'] > 0:
                changes_per_minute = (possession_stats['possession_changes'] / processed_frames) * fps * 60
                print(f"  Possession changes per minute: {changes_per_minute:.1f}")
        else:
            print(f"  No possession data available (teams not assigned or ball not detected sufficiently)")
        
        # Camera movement statistics
        print(f"\nCamera Movement Analysis:")
        print(f"  Camera movement compensation: {'Enabled' if use_camera_movement else 'Disabled'}")
        
        if use_camera_movement and self.camera_movement_per_frame:
            # Calculate camera movement statistics
            movements = self.camera_movement_per_frame
            total_movement = sum(abs(x) + abs(y) for x, y in movements)
            avg_movement_per_frame = total_movement / len(movements) if movements else 0
            max_movement = max(abs(x) + abs(y) for x, y in movements) if movements else 0
            
            # Calculate frames with significant movement
            significant_movement_threshold = 5.0  # pixels
            frames_with_movement = sum(1 for x, y in movements if abs(x) + abs(y) > significant_movement_threshold)
            movement_frame_percentage = (frames_with_movement / len(movements)) * 100 if movements else 0
            
            print(f"  Total camera movement: {total_movement:.2f} pixels")
            print(f"  Average movement per frame: {avg_movement_per_frame:.2f} pixels")
            print(f"  Maximum frame movement: {max_movement:.2f} pixels")
            print(f"  Frames with significant movement: {frames_with_movement}/{len(movements)} ({movement_frame_percentage:.1f}%)")
            print(f"  Camera movement data saved to: {camera_movement_stub_path if camera_movement_stub_path else 'Not saved'}")
            
            # Movement intensity classification
            if avg_movement_per_frame < 1.0:
                movement_intensity = "Very stable (minimal camera movement)"
            elif avg_movement_per_frame < 3.0:
                movement_intensity = "Stable (low camera movement)"
            elif avg_movement_per_frame < 8.0:
                movement_intensity = "Moderate (medium camera movement)"
            elif avg_movement_per_frame < 15.0:
                movement_intensity = "Active (high camera movement)"
            else:
                movement_intensity = "Very active (very high camera movement)"
            
            print(f"  Movement intensity: {movement_intensity}")
        else:
            if use_camera_movement:
                print(f"  Camera movement data: Not available (calculation failed)")
            else:
                print(f"  Camera movement tracking was disabled")
        
        # View transformation statistics
        print(f"\nView Transformation Analysis:")
        print(f"  View transformation: Enabled (pixel to court coordinates)")
        print(f"  Total transformed positions: {transformed_positions_count}")
        
        if tracks:
            # Count successful transformations across all frames
            total_tracks = sum(len(frame_tracks) for frame_tracks in tracks.values())
            successful_transforms = sum(1 for frame_tracks in tracks.values() 
                                    for track in frame_tracks 
                                    if track.get('position_transformed') is not None)
            failed_transforms = total_tracks - successful_transforms
            transform_success_rate = (successful_transforms / total_tracks) * 100 if total_tracks > 0 else 0
            
            print(f"  Successful transformations: {successful_transforms}/{total_tracks} ({transform_success_rate:.1f}%)")
            print(f"  Failed transformations: {failed_transforms} (objects outside court boundaries)")
            print(f"  Court dimensions: 68m x 23.32m")
            print(f"  Pixel vertices: {self.view_transformer.pixel_vertices.tolist()}")
            print(f"  Target vertices: {self.view_transformer.target_vertices.tolist()}")
        
        print(f"\nOutput saved to: {output_path}")
        print(f"Processing complete! {processed_frames} frames processed")
        print(f"="*60)
        
        return processed_frames
    
    def _assign_teams_for_frame(self, frame, frame_tracks):
        """
        Assign team colors for the first frame with enough players.
        
        Args:
            frame: Current frame
            frame_tracks: List of track data for this frame
        """
        if self.team_colors_assigned:
            return
        
        # Get player detections (class 2 after conversion)
        player_detections = {}
        for track in frame_tracks:
            if track['class'] == 2:  # Players (including converted goalkeepers)
                player_detections[track['track_id']] = {
                    'bbox': track['bbox']
                }
        
        # Need at least 4 players to determine teams reliably
        if len(player_detections) >= 4:
            try:
                self.team_assigner.assign_team_color(frame, player_detections)
                self.team_colors_assigned = True
                print(f"Team colors assigned based on {len(player_detections)} players")
            except Exception as e:
                print(f"Error assigning team colors: {e}")
    


    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold set to: {self.confidence_threshold}")
    
    def get_team_assignments(self) -> Dict[int, int]:
        """
        Get current player team assignments.
        
        Returns:
            Dictionary mapping player_id to team_id
        """
        
        return self.team_assigner.player_team_dict.copy()
    
    def get_team_colors(self) -> Dict[int, tuple]:
        """
        Get current team colors in BGR format.
        
        Returns:
            Dictionary mapping team_id to BGR color tuple
        """
        if not self.team_colors_assigned:
            return {}
        
        return {
            1: self.team_assigner.get_team_color_bgr(1),
            2: self.team_assigner.get_team_color_bgr(2)
        }


    def get_current_possession(self) -> Optional[int]:
        """Get the current team in possession."""
        return self.ball_control_tracker.get_current_possession()

    def get_possession_stats(self) -> dict:
        """Get comprehensive possession statistics."""
        # You'll need to track total frames processed
        return self.ball_control_tracker.get_possession_stats(self.total_frames_processed)

    def get_possession_history(self) -> list[Tuple[int, int]]:
        """Get possession change history as list of (frame_idx, team_id) tuples."""
        return self.ball_control_tracker.possession_history.copy()

    def set_possession_parameters(self, distance_threshold: float = None,
                                min_frames: int = None, 
                                smoothing_window: int = None):
        """Update possession tracking parameters."""
        if distance_threshold is not None:
            self.ball_control_tracker.possession_distance_threshold = distance_threshold
        if min_frames is not None:
            self.ball_control_tracker.min_frames_for_possession = min_frames
        if smoothing_window is not None:
            self.ball_control_tracker.possession_smoothing_window = smoothing_window


        def get_camera_movement_stats(self) -> dict:
            """Get camera movement statistics."""
            if self.camera_movement_per_frame is None:
                return {"enabled": False}
            
            movements = self.camera_movement_per_frame
            total_movement = sum(abs(x) + abs(y) for x, y in movements)
            avg_movement = total_movement / len(movements) if movements else 0
            max_movement = max(abs(x) + abs(y) for x, y in movements) if movements else 0
            
            return {
                "enabled": True,
                "total_frames": len(movements),
                "total_movement": total_movement,
                "average_movement_per_frame": avg_movement,
                "maximum_frame_movement": max_movement,
                "movement_data": movements
            }