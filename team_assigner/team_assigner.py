from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)
        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self, frame, bbox):
        # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        x_center, y_center, width, height = bbox
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        
        # Extract player region
        image = frame[y1:y2, x1:x2]
        
        if image.size == 0:
            return np.array([128, 128, 128])  # Default gray color
        
        # Get top half of the image (jersey area)
        top_half_image = image[0:int(image.shape[0]/2), :]
        
        if top_half_image.size == 0:
            return np.array([128, 128, 128])  # Default gray color
        
        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
        # Get the cluster labels for each pixel
        labels = kmeans.labels_
        
        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], 
                          clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for track_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        if len(player_colors) < 2:
            # Not enough players to determine teams
            return
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        if self.kmeans is None:
            return 1  # Default team
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        
        # Special case for player ID 91 (as in original code)
        if player_id == 91:
            team_id = 1
        
        self.player_team_dict[player_id] = team_id
        return team_id
    
    def get_team_color_bgr(self, team_id):
        """Convert team color from RGB to BGR for OpenCV."""
        if team_id in self.team_colors:
            rgb_color = self.team_colors[team_id]
            # Convert RGB to BGR and ensure values are in correct range
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            return bgr_color
        else:
            # Default colors for teams
            team_colors = {
                1: (0, 0, 255),    # Red team
                2: (255, 0, 0)     # Blue team
            }
            return team_colors.get(team_id, (128, 128, 128))  # Gray default