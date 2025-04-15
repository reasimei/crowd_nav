import numpy as np


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def point_to_segment_dist2(p1, p2, p3):
    # Compute distance from point p3 to segment p1-p2
    # p1, p2, p3 are numpy arrays
    line_vec = p2 - p1
    p1_to_p3 = p3 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(p1_to_p3)
    line_unitvec = line_vec / line_len
    proj = np.dot(p1_to_p3, line_unitvec)
    if proj < 0:
        closest_point = p1
    elif proj > line_len:
        closest_point = p2
    else:
        closest_point = p1 + proj * line_unitvec
    dist = np.linalg.norm(p3 - closest_point)
    return dist
