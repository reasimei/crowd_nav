from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])

def limit_speed(vx, vy, max_speed):
    """限制速度 (vx, vy) 不超过 max_speed"""
    mag = (vx**2+vy**2)**0.5# 计算当前速度的模
    if mag > max_speed:
        scale = max_speed / mag  # 计算缩放比例
        vx *= scale
        vy *= scale
    return vx, vy