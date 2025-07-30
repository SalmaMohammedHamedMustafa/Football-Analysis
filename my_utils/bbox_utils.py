def get_center_of_bbox(bbox):
    """Get center coordinates of bounding box."""
    x_center, y_center, width, height = bbox
    return int(x_center), int(y_center)

def get_bbox_width(bbox):
    """Get width of bounding box."""
    return int(bbox[2])