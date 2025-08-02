import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2

class BallControlTracker:
    """
    Tracks ball possession/control for each team based on proximity analysis.
    Determines which team has possession of the ball and maintains possession history.
    """
    
    def __init__(self, possession_distance_threshold: float = 100.0, 
                 min_frames_for_possession: int = 3,
                 possession_smoothing_window: int = 5):
        """
        Initialize the Ball Control Tracker.
        
        Args:
            possession_distance_threshold: Maximum distance for a player to be considered controlling the ball
            min_frames_for_possession: Minimum consecutive frames to confirm possession change
            possession_smoothing_window: Number of frames to use for smoothing possession decisions
        """
        self.possession_distance_threshold = possession_distance_threshold
        self.min_frames_for_possession = min_frames_for_possession
        self.possession_smoothing_window = possession_smoothing_window
        
        # Possession tracking state
        self.current_possession = None  # Current team in possession (1 or 2)
        self.possession_history = []    # List of (frame_idx, team_id) tuples
        self.possession_buffer = []     # Buffer for smoothing possession decisions
        self.frame_possessions = {}     # Dict mapping frame_idx to team_id
        
        # Statistics
        self.team_possession_time = {1: 0, 2: 0}  # Total frames each team had possession
        self.possession_changes = 0     # Number of possession changes
        
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Euclidean distance between centers of two bounding boxes.
        
        Args:
            bbox1: First bounding box [x_center, y_center, width, height]
            bbox2: Second bounding box [x_center, y_center, width, height]
            
        Returns:
            Distance between centers
        """
        x1, y1 = bbox1[0], bbox1[1]
        x2, y2 = bbox2[0], bbox2[1]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _get_closest_player_to_ball(self, ball_bbox: List[float], 
                                   player_tracks: List[Dict], 
                                   team_assignments: Dict[int, int]) -> Optional[Tuple[int, int, float]]:
        """
        Find the closest player to the ball and return their info.
        
        Args:
            ball_bbox: Ball bounding box [x_center, y_center, width, height]
            player_tracks: List of player track dictionaries
            team_assignments: Dictionary mapping player_id to team_id
            
        Returns:
            Tuple of (player_id, team_id, distance) or None if no players found
        """
        closest_player = None
        min_distance = float('inf')
        
        for track in player_tracks:
            if track['class'] == 2:  # Player class
                player_id = track['track_id']
                player_bbox = track['bbox']
                
                # Get team assignment for this player
                team_id = team_assignments.get(player_id, 1)  # Default to team 1
                
                # Calculate distance to ball
                distance = self._calculate_distance(ball_bbox, player_bbox)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_player = (player_id, team_id, distance)
        
        return closest_player
    
    def _smooth_possession_decision(self, raw_possession: Optional[int]) -> Optional[int]:
        """
        Apply smoothing to possession decisions to reduce flickering.
        
        Args:
            raw_possession: Raw possession decision for current frame
            
        Returns:
            Smoothed possession decision
        """
        # Add current decision to buffer
        self.possession_buffer.append(raw_possession)
        
        # Keep buffer size limited
        if len(self.possession_buffer) > self.possession_smoothing_window:
            self.possession_buffer.pop(0)
        
        # If buffer isn't full yet, return current possession
        if len(self.possession_buffer) < self.min_frames_for_possession:
            return self.current_possession
        
        # Count votes for each team in the buffer
        team_votes = {1: 0, 2: 0, None: 0}
        for possession in self.possession_buffer[-self.min_frames_for_possession:]:
            team_votes[possession] += 1
        
        # Find team with most votes
        max_votes = max(team_votes.values())
        winning_teams = [team for team, votes in team_votes.items() if votes == max_votes]
        
        # If there's a clear winner and it's not None, update possession
        if len(winning_teams) == 1 and winning_teams[0] is not None:
            return winning_teams[0]
        
        # Otherwise, keep current possession
        return self.current_possession
    
    def update_ball_control(self, frame_idx: int, ball_bbox: Optional[List[float]], 
                           player_tracks: List[Dict], team_assignments: Dict[int, int]) -> Optional[int]:
        """
        Update ball control analysis for the current frame.
        
        Args:
            frame_idx: Current frame index
            ball_bbox: Ball bounding box [x_center, y_center, width, height] or None if no ball
            player_tracks: List of player track dictionaries
            team_assignments: Dictionary mapping player_id to team_id
            
        Returns:
            Team ID (1 or 2) that has possession, or None if undetermined
        """
        raw_possession = None
        
        if ball_bbox is not None and player_tracks:
            # Find closest player to ball
            closest_info = self._get_closest_player_to_ball(ball_bbox, player_tracks, team_assignments)
            
            if closest_info:
                player_id, team_id, distance = closest_info
                
                # Check if player is close enough to have possession
                if distance <= self.possession_distance_threshold:
                    raw_possession = team_id
        
        # Apply smoothing to reduce flickering
        smoothed_possession = self._smooth_possession_decision(raw_possession)
        
        # Update possession if it changed
        if smoothed_possession != self.current_possession:
            if self.current_possession is not None and smoothed_possession is not None:
                self.possession_changes += 1
            
            self.current_possession = smoothed_possession
            
            if smoothed_possession is not None:
                self.possession_history.append((frame_idx, smoothed_possession))
        
        # Record possession for this frame
        self.frame_possessions[frame_idx] = self.current_possession
        
        # Update possession time statistics
        if self.current_possession is not None:
            self.team_possession_time[self.current_possession] += 1
        
        return self.current_possession
    
    def get_possession_stats(self, total_frames: int) -> Dict:
        """
        Get comprehensive possession statistics.
        
        Args:
            total_frames: Total number of frames processed
            
        Returns:
            Dictionary containing possession statistics
        """
        total_possession_frames = sum(self.team_possession_time.values())
        
        stats = {
            'total_frames': total_frames,
            'frames_with_possession': total_possession_frames,
            'possession_percentage': (total_possession_frames / total_frames * 100) if total_frames > 0 else 0,
            'team_1_frames': self.team_possession_time[1],
            'team_2_frames': self.team_possession_time[2],
            'team_1_percentage': (self.team_possession_time[1] / total_possession_frames * 100) if total_possession_frames > 0 else 0,
            'team_2_percentage': (self.team_possession_time[2] / total_possession_frames * 100) if total_possession_frames > 0 else 0,
            'possession_changes': self.possession_changes,
            'average_possession_duration': total_possession_frames / max(self.possession_changes, 1)
        }
        
        return stats
    
    def get_current_possession(self) -> Optional[int]:
        """Get the current team in possession."""
        return self.current_possession
    
    def get_possession_for_frame(self, frame_idx: int) -> Optional[int]:
        """Get possession for a specific frame."""
        return self.frame_possessions.get(frame_idx)
    
    def draw_possession_indicator(self, frame: np.ndarray, 
                                 possession_team: Optional[int],
                                 team_colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
        """
        Draw possession indicator on the frame.
        
        Args:
            frame: Input frame
            possession_team: Team ID that has possession (1 or 2) or None
            team_colors: Dictionary mapping team_id to BGR color tuple
            
        Returns:
            Frame with possession indicator drawn
        """
        if possession_team is None:
            return frame
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Possession indicator position (top-left corner)
        indicator_x = 30
        indicator_y = 50
        indicator_width = 200
        indicator_height = 40
        
        # Get team color
        team_color = team_colors.get(possession_team, (128, 128, 128))
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (indicator_x - 10, indicator_y - 30), 
                     (indicator_x + indicator_width + 10, indicator_y + 10),
                     (0, 0, 0), -1)  # Black background
        
        # Draw colored rectangle for possession
        cv2.rectangle(frame,
                     (indicator_x, indicator_y - 25),
                     (indicator_x + indicator_width, indicator_y + 5),
                     team_color, -1)
        
        # Add possession text
        text = f"POSSESSION: TEAM {possession_team}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Calculate text position to center it
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = indicator_x + (indicator_width - text_width) // 2
        text_y = indicator_y - 5
        
        # Draw text with contrasting color
        text_color = (255, 255, 255) if sum(team_color) < 384 else (0, 0, 0)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        return frame
    
    def reset(self):
        """Reset all tracking state."""
        self.current_possession = None
        self.possession_history.clear()
        self.possession_buffer.clear()
        self.frame_possessions.clear()
        self.team_possession_time = {1: 0, 2: 0}
        self.possession_changes = 0