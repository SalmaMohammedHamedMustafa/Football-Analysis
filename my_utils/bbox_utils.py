def get_center_of_bbox(bbox):
    """Get center coordinates of bounding box."""
    x_center, y_center, width, height = bbox
    return int(x_center), int(y_center)

def get_bbox_width(bbox):
    """Get width of bounding box."""
    return int(bbox[2])

def get_bbox_height(bbox):
    """Get height of bounding box."""
    return int(bbox[3])

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)