import numpy as np
import json

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

def hose_model(robot1, robot2, hose_length):
    """
    生成软管形状的函数。

    参数：
        robot1: 第一个机器人的位置 (x1, y1)
        robot2: 第二个机器人的位置 (x2, y2)
        hose_length: 软管的长度

    返回：
        x, y: 表示软管曲线的点集
    """
    x1, y1 = robot1
    x2, y2 = robot2

    # 计算两机器人之间的距离
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if distance >= hose_length:
        # 如果距离>=软管长度，画直线
        x = np.linspace(x1, x2, 100)
        y = np.linspace(y1, y2, 100)
    else:
        # 如果距离<软管长度，画正弦曲线
        num_points = 100
        x = np.linspace(x1, x2, num_points)
        wavelength = 2 * distance  # 半个波长为机器人间距
        amplitude = np.sqrt((hose_length / 2)**2 - (distance / 2)**2)
        phase = np.arctan2(y2 - y1, x2 - x1)  # 曲线方向角度

        # 在局部坐标系中生成正弦曲线
        local_x = np.linspace(0, distance, num_points)
        if (x1>0 and x2>0 and (y1<0 or y2<0)) or (y1<0 and y2<0):
            local_y = -amplitude * np.sin(2 * np.pi * local_x / wavelength)
        else:
            local_y = amplitude * np.sin(2 * np.pi * local_x / wavelength)

        # 将局部坐标系转换回全局坐标系
        rotation_matrix = np.array([[np.cos(phase), -np.sin(phase)],
                                     [np.sin(phase), np.cos(phase)]])
        global_coords = rotation_matrix @ np.array([local_x, local_y])
        x, y = global_coords[0] + x1, global_coords[1] + y1

    return x, y

def point_to_hose_curve(point, robot1, robot2, hose_length, num_points=100):
    """
    计算一个点到软管曲线的最短距离
    
    参数：
        point: 给定点的坐标 (px, py)
        robot1: 第一个机器人的位置 (x1, y1)
        robot2: 第二个机器人的位置 (x2, y2)
        hose_length: 软管的长度
        num_points: 用于计算软管曲线的点的数量（默认为100）
    
    返回：
        最短距离
    """
    # 获取软管曲线的坐标
    x, y = hose_model(robot1, robot2, hose_length)
    
    # 计算点到软管曲线各点的距离
    distances = np.sqrt((x - point[0])**2 + (y - point[1])**2)
    
    # 返回最小距离
    return np.min(distances)

def fix_incomplete_json(json_str):
        """自动修复缺少闭合括号的 JSON 字符串"""
        # 统计大括号数量
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        # 如果缺少闭合括号
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            json_str += '}' * missing_braces  # 补充缺失的闭合括号
        
        # 尝试解析验证
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # 如果仍然无效，尝试更智能的修复
            return json_str + '}' * (open_braces - close_braces)  # 强制补全