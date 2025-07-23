import cv2
import ultralytics
import numpy as np

class FootballVideoTracker:
    """
    Detect, track objects, draw custom ellipses on each,
    store tracking data, and print accurate counts of detected classes.
    """
    def __init__(self, model_path, input_video, output_video, confidence_threshold=0.5):
        self.model = ultralytics.YOLO(model_path)
        self.cap = cv2.VideoCapture(input_video)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))
        
        self.confidence_threshold = confidence_threshold

        self.class_colors = {
            'ball': (0, 255, 255),
            'goalkeeper': (0, 0, 255),
            'player': (255, 0, 0),
            'referee': (0, 0, 0),
        }

        # Store tracking info: track_id -> info with 'class_name', 'positions', 'confidences'
        self.tracking_data = {}
        
        # Track maximum counts per frame for accurate object counting
        self.max_players_in_frame = 0
        self.max_balls_in_frame = 0
        self.max_referees_in_frame = 0
        
        # Track frame-by-frame counts for analysis
        self.frame_counts = {
            'players': [],
            'balls': [],
            'referees': []
        }

    def process(self):
        frame_number = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_number += 1
            
            # Use tracker with persist=True to maintain consistent IDs
            results = self.model.track(source=frame, persist=True, verbose=False)
            
            # Count objects in current frame
            current_players = 0
            current_balls = 0
            current_referees = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Apply confidence threshold filter
                        conf = float(box.conf[0])
                        if conf < self.confidence_threshold:
                            continue
                            
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int(y2)

                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        original_class = class_name

                        # Count objects in current frame
                        if class_name in ['player', 'goalkeeper']:
                            current_players += 1
                            # Convert goalkeeper to player for display
                            if class_name == 'goalkeeper':
                                class_name = 'player'
                        elif class_name == 'ball':
                            current_balls += 1
                        elif class_name == 'referee':
                            current_referees += 1

                        color = self.class_colors.get(class_name, (128, 128, 128))
                        box_width = x2 - x1

                        # Ellipse size based on class
                        if class_name == 'ball':
                            ellipse_width = max(int(box_width * 0.5), 6)
                            ellipse_height = max(int(ellipse_width * 0.8), 5)
                        else:
                            ellipse_width = max(int(box_width * 0.3), 10)
                            ellipse_height = max(int(ellipse_width * 0.4), 4)

                        # Draw filled ellipse and border
                        cv2.ellipse(frame, (center_x, center_y), (ellipse_width, ellipse_height), 0, 0, 360, color, -1)
                        cv2.ellipse(frame, (center_x, center_y), (ellipse_width + 2, ellipse_height + 1), 0, 0, 360, (0, 0, 0), 2)

                        # Draw label
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (center_x - 30, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                        # Store tracking data (keeping original functionality)
                        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                        if track_id is not None:
                            if track_id not in self.tracking_data:
                                self.tracking_data[track_id] = {
                                    'class_name': class_name,
                                    'original_class': original_class,
                                    'positions': [],
                                    'confidences': [],
                                }
                            self.tracking_data[track_id]['positions'].append((center_x, center_y))
                            self.tracking_data[track_id]['confidences'].append(conf)

            # Update maximum counts
            self.max_players_in_frame = max(self.max_players_in_frame, current_players)
            self.max_balls_in_frame = max(self.max_balls_in_frame, current_balls)
            self.max_referees_in_frame = max(self.max_referees_in_frame, current_referees)
            
            # Store frame counts for analysis
            self.frame_counts['players'].append(current_players)
            self.frame_counts['balls'].append(current_balls)
            self.frame_counts['referees'].append(current_referees)
            
            # Optional: Display current frame counts on video
            count_text = f"Frame {frame_number} - Players: {current_players}, Balls: {current_balls}, Referees: {current_referees}"
            cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write annotated frame to output video
            self.out.write(frame)

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing complete! Check output video.")

        # Print accurate statistics
        self.print_statistics()

        return self.tracking_data

    def print_statistics(self):
        """Print detailed statistics about detected objects"""
        print("\n" + "="*60)
        print("OBJECT DETECTION STATISTICS")
        print("="*60)
        
        print("\n--- ACCURATE OBJECT COUNTS (Based on simultaneous detections) ---")
        print(f"Maximum players detected simultaneously: {self.max_players_in_frame}")
        print(f"Maximum balls detected simultaneously: {self.max_balls_in_frame}")
        print(f"Maximum referees detected simultaneously: {self.max_referees_in_frame}")
        
        # Calculate average counts
        if self.frame_counts['players']:
            avg_players = sum(self.frame_counts['players']) / len(self.frame_counts['players'])
            avg_balls = sum(self.frame_counts['balls']) / len(self.frame_counts['balls'])
            avg_referees = sum(self.frame_counts['referees']) / len(self.frame_counts['referees'])
            
            print(f"\n--- AVERAGE OBJECTS PER FRAME ---")
            print(f"Average players per frame: {avg_players:.1f}")
            print(f"Average balls per frame: {avg_balls:.1f}")
            print(f"Average referees per frame: {avg_referees:.1f}")
        
        print(f"\n--- TRACKING STATISTICS (Total track IDs created) ---")
        # Count unique detected objects by class from tracking data
        track_count_players = sum(1 for v in self.tracking_data.values() if v['class_name'] == 'player')
        track_count_balls = sum(1 for v in self.tracking_data.values() if v['class_name'] == 'ball')
        track_count_referees = sum(1 for v in self.tracking_data.values() if v['class_name'] == 'referee')
        
        print(f"Total player track IDs: {track_count_players}")
        print(f"Total ball track IDs: {track_count_balls}")
        print(f"Total referee track IDs: {track_count_referees}")
        print("(Note: High track ID counts indicate objects being re-tracked)")
        
        # Calculate tracking efficiency
        if self.max_players_in_frame > 0:
            player_efficiency = self.max_players_in_frame / max(track_count_players, 1)
            print(f"\nPlayer tracking efficiency: {player_efficiency:.2%}")
            print(f"(Ratio of actual players to track IDs - higher is better)")



if __name__ == "__main__":
    # Adjust model path and video paths as needed
    # You can also adjust confidence_threshold (default 0.5)
    processor = FootballVideoTracker("best.pt", "input.mp4", "output.mp4", confidence_threshold=0.5)
    tracking_info = processor.process()
    
