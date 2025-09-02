import os
import cv2
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import ultralytics
from ultralytics import YOLO
import sys
from sklearn.cluster import KMeans
sys.path.append('../')
from team_assigner import TeamAssigner
from my_utils import measure_distance, measure_xy_distance
from ball_control_tracker import BallControlTracker


class CameraMovementEstimator:

    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[0:20, :] = 1  # Top horizontal strip (first 20 rows)
        mask_features[900:1050, :] = 1  # Bottom horizontal strip (rows 900 to 1050, all columns)

        
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )
    
    def get_camera_movement_streaming(self, video_path, read_from_stub=False, stub_path=None):
        """
        Get camera movement using streaming approach for memory efficiency.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        camera_movement = []
        frame_count = 0
        old_gray = None
        old_features = None
        
        print("Calculating camera movement...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if frame_count == 0:
                # First frame
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                camera_movement.append([0, 0])
            else:
                # Calculate optical flow
                if old_features is not None and len(old_features) > 0:
                    new_features, status, error = cv2.calcOpticalFlowPyrLK(
                        old_gray, frame_gray, old_features, None, **self.lk_params
                    )
                    
                    max_distance = 0
                    camera_movement_x, camera_movement_y = 0, 0
                    
                    # Find the maximum movement
                    for i, (new, old) in enumerate(zip(new_features, old_features)):
                        if status[i]:
                            new_point = new.ravel()
                            old_point = old.ravel()
                            distance = measure_distance(new_point, old_point)
                            
                            if distance > max_distance:
                                max_distance = distance
                                camera_movement_x, camera_movement_y = measure_xy_distance(
                                    old_point, new_point
                                )
                    
                    if max_distance > self.minimum_distance:
                        camera_movement.append([camera_movement_x, camera_movement_y])
                        # Update features
                        old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    else:
                        camera_movement.append([0, 0])
                else:
                    camera_movement.append([0, 0])
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                
                old_gray = frame_gray.copy()
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"   Camera movement: processed {frame_count} frames")
        
        cap.release()
        
        # Save to stub if path provided
        if stub_path is not None:
            stub_path = Path(stub_path)
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            print(f"Camera movement saved to: {stub_path}")
        
        print(f"Camera movement calculation complete: {len(camera_movement)} frames")
        return camera_movement
    
    def adjust_position_for_camera_movement(self, position, camera_movement):
        """
        Adjust a single position for camera movement.
        
        Args:
            position: (x, y) position tuple
            camera_movement: [x_movement, y_movement] for the frame
            
        Returns:
            Adjusted (x, y) position tuple
        """
        return (
            position[0] - camera_movement[0],
            position[1] - camera_movement[1]
        )
    
    def adjust_bbox_for_camera_movement(self, bbox, camera_movement):
        """
        Adjust a bounding box for camera movement.
        
        Args:
            bbox: [x_center, y_center, width, height] bounding box
            camera_movement: [x_movement, y_movement] for the frame
            
        Returns:
            Adjusted bounding box
        """
        return [
            bbox[0] - camera_movement[0],  # x_center
            bbox[1] - camera_movement[1],  # y_center
            bbox[2],                       # width (unchanged)
            bbox[3]                        # height (unchanged)
        ]
    
    def draw_camera_movement(self, frame, camera_movement, frame_idx):
        """
        Draw camera movement information on a single frame.
        
        Args:
            frame: Input frame
            camera_movement: List of camera movements for all frames
            frame_idx: Current frame index
            
        Returns:
            Frame with camera movement overlay
        """
        if frame_idx >= len(camera_movement):
            return frame
        
        frame = frame.copy()
        overlay = frame.copy()
        
        # Create semi-transparent background
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Get movement for current frame
        x_movement, y_movement = camera_movement[frame_idx]
        
        # Add text
        frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        return frame