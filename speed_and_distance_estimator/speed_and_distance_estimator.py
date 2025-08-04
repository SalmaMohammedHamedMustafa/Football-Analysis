import cv2
import numpy as np
import sys 
sys.path.append('../')
from my_utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5  # Frames to look back for speed calculation
        self.frame_rate = 24   # Will be updated with actual video FPS
        self.max_reasonable_speed = 28.0  # km/h - realistic but not too restrictive
        self.min_distance_threshold = 0.1  # meters - reduced back to catch more movement
        self.min_frame_gap = 2  # Reduced to allow more frequent calculations
        
    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Add speed and distance calculations to tracks.
        Enhanced version with better filtering and smoothing.
        """
        print(f"Calculating speed and distance with frame rate: {self.frame_rate} FPS")
        
        # Track cumulative distances and position history for each object
        total_distance = {}
        position_history = {}
        last_speed_calculation = {}  # Track when we last calculated speed for smoothing
        
        if not tracks:
            return
        
        # Get all frame numbers and sort them
        frame_numbers = sorted(tracks.keys())
        total_frames = len(frame_numbers)
        
        processed_objects = set()
        speed_calculations = 0
        valid_speeds = 0
        filtered_speeds = 0
        noise_filtered = 0
        
        # Process each frame sequentially
        for i, frame_num in enumerate(frame_numbers):
            if frame_num not in tracks:
                continue
                
            # Process each track in the current frame
            for track in tracks[frame_num]:
                track_id = track['track_id']
                class_id = track['class']
                
                processed_objects.add((track_id, class_id))
                
                # Initialize tracking data for new objects
                if track_id not in total_distance:
                    total_distance[track_id] = 0.0
                    position_history[track_id] = []
                    last_speed_calculation[track_id] = -999
                
                # Get current position - prefer transformed coordinates
                current_position = None
                coordinate_system = "none"
                
                # Priority order: transformed -> adjusted -> original
                if track.get('position_transformed') is not None:
                    current_position = track['position_transformed']
                    coordinate_system = "transformed"
                elif track.get('position_adjusted') is not None:
                    current_position = track['position_adjusted']
                    coordinate_system = "adjusted"
                else:
                    # Fallback to bbox center
                    from my_utils import get_center_of_bbox
                    bbox = track.get('bbox')
                    if bbox:
                        current_position = list(get_center_of_bbox(bbox))
                        coordinate_system = "pixel"
                
                # Initialize values
                track['speed'] = None
                track['distance'] = total_distance[track_id]
                track['coordinate_system_used'] = coordinate_system
                
                if current_position is None:
                    continue
                
                # Add current position to history
                position_history[track_id].append({
                    'frame': frame_num,
                    'position': current_position,
                    'coordinate_system': coordinate_system
                })
                
                # Keep only recent positions within frame window
                cutoff_frame = frame_num - self.frame_window
                position_history[track_id] = [
                    p for p in position_history[track_id] 
                    if p['frame'] >= cutoff_frame
                ]
                
                # Calculate speed only if enough time has passed since last calculation
                frames_since_last = frame_num - last_speed_calculation[track_id]
                
                if (len(position_history[track_id]) >= 2 and 
                    frames_since_last >= self.min_frame_gap):
                    
                    speed, distance_increment = self._calculate_speed_from_history(
                        position_history[track_id], frame_num, class_id
                    )
                    
                    speed_calculations += 1
                    
                    if speed is not None and speed > 0:
                        # Apply class-specific speed filters
                        max_speed = self._get_max_speed_for_class(class_id)
                        
                        if speed <= max_speed:
                            track['speed'] = speed
                            total_distance[track_id] += distance_increment
                            track['distance'] = total_distance[track_id]
                            last_speed_calculation[track_id] = frame_num
                            valid_speeds += 1
                        else:
                            # Speed too high, likely tracking error
                            track['speed'] = None
                            filtered_speeds += 1
                    elif speed == 0.0:
                        # Noise filtered out
                        track['speed'] = 0.0
                        noise_filtered += 1
                    else:
                        track['speed'] = None
        
        print(f"Enhanced speed calculation summary:")
        print(f"  Objects processed: {len(processed_objects)}")
        print(f"  Speed calculations attempted: {speed_calculations}")
        print(f"  Valid speeds calculated: {valid_speeds}")
        print(f"  Speeds filtered (too high): {filtered_speeds}")
        print(f"  Noise movements filtered: {noise_filtered}")
        print(f"  Min frame gap: {self.min_frame_gap} frames")
        print(f"  Min distance threshold: {self.min_distance_threshold} meters")
    
    def _get_max_speed_for_class(self, class_id):
        """Get maximum reasonable speed for different object classes."""
        speed_limits = {
            0: 80.0,   # Ball - can move very fast
            1: 28.0,   # Goalkeeper - slightly more generous
            2: 28.0,   # Player - professional soccer players can reach this in sprints
            3: 22.0    # Referee - typically slower but can run
        }
        return speed_limits.get(class_id, self.max_reasonable_speed)
    
    def _calculate_speed_from_history(self, position_history, current_frame, class_id):
        """
        Calculate speed using position history with enhanced filtering and smoothing.
        
        Args:
            position_history: List of position records
            current_frame: Current frame number
            class_id: Object class for additional filtering
            
        Returns:
            Tuple of (speed_km_h, distance_increment)
        """
        if len(position_history) < 2:
            return None, 0.0
        
        # Get current position
        current_record = position_history[-1]
        
        # Try to find the best reference frame for calculation
        # Balance between stability and capturing movement
        best_reference = None
        best_frame_gap = 0
        
        for record in reversed(position_history[:-1]):
            frame_diff = current_frame - record['frame']
            
            # More flexible frame gap selection
            if (self.min_frame_gap <= frame_diff <= self.frame_window and 
                record['coordinate_system'] == current_record['coordinate_system']):
                # Prefer moderate frame gaps (2-4 frames) for balance
                if frame_diff >= 2 and (best_reference is None or frame_diff > best_frame_gap):
                    best_reference = record
                    best_frame_gap = frame_diff
        
        if best_reference is None:
            return None, 0.0
        
        # Calculate distance and time
        try:
            distance_meters = measure_distance(
                best_reference['position'], 
                current_record['position']
            )
            
            # Enhanced noise filtering
            if distance_meters < self.min_distance_threshold:
                return 0.0, 0.0
            
            frame_diff = current_record['frame'] - best_reference['frame']
            time_elapsed = frame_diff / self.frame_rate
            
            if time_elapsed <= 0:
                return None, 0.0
            
            # Calculate speed
            speed_ms = distance_meters / time_elapsed
            speed_kmh = speed_ms * 3.6
            
            # Additional filtering based on coordinate system and class
            if not self._is_realistic_speed(speed_kmh, distance_meters, time_elapsed, 
                                          current_record['coordinate_system'], class_id):
                return None, 0.0
            
            return speed_kmh, distance_meters
            
        except Exception as e:
            print(f"Error calculating speed: {e}")
            return None, 0.0
    
    def _is_realistic_speed(self, speed_kmh, distance_meters, time_elapsed, 
                           coordinate_system, class_id):
        """
        Balanced realism check for calculated speeds.
        Less aggressive filtering to capture more legitimate movement.
        """
        # Basic speed limit check
        max_speed = self._get_max_speed_for_class(class_id)
        if speed_kmh > max_speed:
            return False
        
        # More lenient acceleration check
        # Allow higher acceleration for short bursts (soccer involves quick direction changes)
        max_acceleration = 8.0 if class_id in [1, 2] else 12.0  # m/s² - more generous
        speed_ms = speed_kmh / 3.6
        
        # Only check acceleration for longer time periods
        if time_elapsed > 0.5 and speed_ms / time_elapsed > max_acceleration:
            return False
        
        # Coordinate system specific checks - more lenient
        if coordinate_system == "pixel":
            # For pixel coordinates, allow more movement
            if distance_meters > 200 and time_elapsed < 0.5:  # More generous pixel distance
                return False
        
        elif coordinate_system == "transformed":
            # For court coordinates, allow realistic soccer movement
            # Top soccer players can move ~8-10 m/s in short bursts
            max_distance_per_second = 35.0  # meters - more generous
            if distance_meters / time_elapsed > max_distance_per_second:
                return False
        
        # More lenient minimum speed check
        if speed_kmh < 0.3 and class_id in [1, 2]:  # Very small movements might be real
            return True  # Allow small movements
        
        return True
    
    def draw_speed_and_distance(self, frame, frame_tracks, font_scale=0.5):
        """
        Draw speed and distance on a single frame with improved formatting.
        """
        annotated_frame = frame.copy()
        
        for track in frame_tracks:
            # Check if speed and distance data exists
            speed = track.get('speed')
            distance = track.get('distance')
            coordinate_system = track.get('coordinate_system_used', 'unknown')
            
            # Skip if no data available
            if speed is None and distance is None:
                continue
            
            # Only show for players to avoid clutter
            if track['class'] != 2:  # Only show for players
                continue
            
            # Get foot position for text placement
            bbox = track['bbox']
            foot_position = get_foot_position(bbox)
            
            # Adjust position for text (below the player)
            text_x = int(foot_position[0])
            text_y = int(foot_position[1] + 40)
            
            # Ensure text is within frame bounds
            frame_height, frame_width = frame.shape[:2]
            text_x = max(10, min(text_x, frame_width - 200))
            text_y = max(50, min(text_y, frame_height - 50))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            
            # Prepare text lines
            text_lines = []
            
            if speed is not None and speed > 0:
                text_lines.append(f"{speed:.1f} km/h")
            
            if distance is not None and distance > 0:
                text_lines.append(f"{distance:.1f}m")
            
            # Add coordinate system indicator for debugging (small text)
            if coordinate_system != 'unknown':
                coord_indicator = coordinate_system[0].upper()  # T, A, or P
                text_lines.append(f"({coord_indicator})")
            
            # Draw each text line with background
            for i, text in enumerate(text_lines):
                y_offset = i * 22
                current_y = text_y + y_offset
                
                # Get text dimensions for background
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(annotated_frame, 
                             (text_x - 2, current_y - text_h - 2), 
                             (text_x + text_w + 2, current_y + 2), 
                             (255, 255, 255), -1)
                
                # Draw text
                text_color = (0, 0, 0) if i < 2 else (128, 128, 128)  # Gray for coordinate system
                font_scale_current = font_scale if i < 2 else font_scale * 0.7  # Smaller for coordinate system
                
                cv2.putText(annotated_frame, text, (text_x, current_y), 
                           font, font_scale_current, text_color, thickness if i < 2 else 1)
        
        return annotated_frame

    def get_speed_statistics(self, tracks):
        """
        Get comprehensive speed statistics from tracks with enhanced analysis.
        
        Returns:
            Dictionary with detailed speed statistics
        """
        all_speeds = []
        player_speeds = []
        ball_speeds = []
        referee_speeds = []
        
        coordinate_system_counts = {'transformed': 0, 'adjusted': 0, 'pixel': 0}
        speed_categories = {'stationary': 0, 'walking': 0, 'jogging': 0, 'running': 0, 'sprinting': 0}
        
        for frame_tracks in tracks.values():
            for track in frame_tracks:
                speed = track.get('speed')
                if speed is not None and speed >= 0:  # Include zero speeds
                    all_speeds.append(speed)
                    
                    class_id = track.get('class', 0)
                    if class_id == 0:  # Ball
                        ball_speeds.append(speed)
                    elif class_id in [1, 2]:  # Players/goalkeepers
                        player_speeds.append(speed)
                        
                        # Categorize player speeds more realistically
                        if speed < 1.0:
                            speed_categories['stationary'] += 1
                        elif speed < 5.0:
                            speed_categories['walking'] += 1
                        elif speed < 10.0:
                            speed_categories['jogging'] += 1
                        elif speed < 18.0:
                            speed_categories['running'] += 1
                        else:
                            speed_categories['sprinting'] += 1
                            
                    elif class_id == 3:  # Referees
                        referee_speeds.append(speed)
                    
                    # Count coordinate systems used
                    coord_sys = track.get('coordinate_system_used', 'unknown')
                    if coord_sys in coordinate_system_counts:
                        coordinate_system_counts[coord_sys] += 1
        
        # Calculate percentages for speed categories
        total_player_measurements = sum(speed_categories.values())
        speed_category_percentages = {}
        if total_player_measurements > 0:
            for category, count in speed_categories.items():
                speed_category_percentages[category] = (count / total_player_measurements) * 100
        
        stats = {
            'total_measurements': len(all_speeds),
            'coordinate_system_usage': coordinate_system_counts,
            'speed_categories': speed_categories,
            'speed_category_percentages': speed_category_percentages,
            'all_speeds': {
                'count': len(all_speeds),
                'mean': np.mean(all_speeds) if all_speeds else 0,
                'max': np.max(all_speeds) if all_speeds else 0,
                'min': np.min(all_speeds) if all_speeds else 0,
                'std': np.std(all_speeds) if all_speeds else 0,
                'percentile_95': np.percentile(all_speeds, 95) if all_speeds else 0,
                'percentile_99': np.percentile(all_speeds, 99) if all_speeds else 0,
                'median': np.median(all_speeds) if all_speeds else 0
            },
            'player_speeds': {
                'count': len(player_speeds),
                'mean': np.mean(player_speeds) if player_speeds else 0,
                'max': np.max(player_speeds) if player_speeds else 0,
                'std': np.std(player_speeds) if player_speeds else 0,
                'median': np.median(player_speeds) if player_speeds else 0
            },
            'ball_speeds': {
                'count': len(ball_speeds),
                'mean': np.mean(ball_speeds) if ball_speeds else 0,
                'max': np.max(ball_speeds) if ball_speeds else 0
            },
            'referee_speeds': {
                'count': len(referee_speeds),
                'mean': np.mean(referee_speeds) if referee_speeds else 0,
                'max': np.max(referee_speeds) if referee_speeds else 0
            }
        }
        
        return stats