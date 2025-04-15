from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state import FullState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import numpy as np
import logging

# robot.py

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index
        self.env = None  # Will be set later

    def set_env(self, env):
        self.env = env

    # def get_other_robot_state(self):
    #     if self.robot_index == 0:
    #         return self.env.robot2.get_full_state()
    #     else:
    #         return self.env.robot1.get_full_state()

    def get_other_robot_state(self):
        # return self.env.robots[robot_index].get_full_state()
        # other_robots_num = 0
        other_robots_states = []
        for i, robot in enumerate(self.env.robots):
            if i != self.robot_index:
                other_robots_states.append(robot.get_full_state())
        return other_robots_states

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        other_robot_states = []
        other_robot_states = self.get_other_robot_state()
        # print(other_robot_states[0])
        state = JointState(self.get_full_state(), other_robot_states, ob)
        action = self.policy.predict(state)
        return action

    def compute_position(self, action, time_step):
        """
        Compute future position based on action and timestep
        
        Args:
            action: ActionXY or ActionRot object containing velocity/rotation commands
            time_step: Time duration to predict forward
            
        Returns:
            tuple: (x, y) predicted position
        """
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * time_step 
            py = self.py + action.vy * time_step
        else:
            # For non-holonomic, use current heading + rotation
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * time_step
            py = self.py + np.sin(theta) * action.v * time_step
            
        return (px, py)

    def act_avoid_humans(self, ob):
        """
        Conservative action selection in human zone - slow speed or wait
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        try:
            # Check if there are humans nearby
            humans_nearby = False
            for human_state in ob:
                if not isinstance(human_state, Robot):
                    dist = np.linalg.norm(np.array([self.px - human_state.px, self.py - human_state.py]))
                    if dist < 3.0:  # Conservative safety distance
                        humans_nearby = True
                        break
            
            if humans_nearby:
                # If humans are nearby, either wait or move very slowly
                if self.kinematics == 'holonomic':
                    # Almost stop, with minimal movement towards goal
                    direction = np.array([self.gx - self.px, self.gy - self.py])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    return ActionXY(direction[0] * self.v_pref * 0.1, direction[1] * self.v_pref * 0.1)
                else:
                    return ActionRot(0, 0)  # Stop for non-holonomic robot
            
            # Create joint state and get prediction
            state = JointState(self.get_full_state(), [], ob)
            action = self.policy.predict(state)
            
            # Validate action
            if action is None:
                # Fallback to safe action
                direction = np.array([self.gx - self.px, self.gy - self.py])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                return ActionXY(direction[0] * self.v_pref * 0.5, direction[1] * self.v_pref * 0.5)
                
            return action
            
        except Exception as e:
            logging.error(f"Error in act_avoid_humans: {e}")
            # Return safe default action
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)


    def act_avoid_robots(self, ob):
        """
        机器人避让策略,考虑:
        1. 与其他机器人的避碰
        2. 与软管的避碰
        3. 与静态human的避碰
        4. 根据位置设置不同运动策略
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        # 安全距离参数
        robot_safe_dist = 2.0  # 机器人间安全距离
        hose_safe_dist = 1.5   # 与软管安全距离
        human_safe_dist = 1.0  # 与静态human安全距离
        
        # 获取当前位置和目标
        cur_pos = np.array([self.px, self.py])
        goal_pos = np.array([self.gx, self.gy])
        
        # 根据机器人在组中的位置确定运动策略
        group_pos = self.robot_index % 8
        radius = self.env.circle_radius - (self.robot_index // 8) * 2

        # 计算期望运动方向
        if group_pos < 2:  # 下部机器人
            # 需要绕过中心区域向上运动
            if self.py < 0:
                # 先向左或右移动以避开中心
                desired_pos = np.array([self.px + (-1 if self.px > 0 else 1) * 2, self.py])
            else:
                # 然后向上移动到目标
                desired_pos = goal_pos
        elif group_pos < 4:  # 右侧机器人
            if self.px > 0:
                # 先向上或下移动
                desired_pos = np.array([self.px, self.py + (-1 if self.py > 0 else 1) * 2])
            else:
                desired_pos = goal_pos
        elif group_pos < 6:  # 上部机器人
            if self.py > 0:
                desired_pos = np.array([self.px + (-1 if self.px > 0 else 1) * 2, self.py])
            else:
                desired_pos = goal_pos
        else:  # 左侧机器人
            if self.px < 0:
                desired_pos = np.array([self.px, self.py + (-1 if self.py > 0 else 1) * 2])
            else:
                desired_pos = goal_pos

        # 计算期望速度方向
        direction = desired_pos - cur_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction = direction / dist

        # 检查避碰约束
        need_to_avoid = False
        modified_direction = direction.copy()

        # 1. 检查与其他机器人的碰撞
        for i, robot in enumerate(self.env.robots):
            if i != self.robot_index:
                robot_pos = np.array([robot.px, robot.py])
                robot_dist = np.linalg.norm(robot_pos - cur_pos)
                if robot_dist < robot_safe_dist:
                    avoid_direction = (cur_pos - robot_pos) / robot_dist
                    modified_direction += avoid_direction * (robot_safe_dist - robot_dist) / robot_safe_dist
                    need_to_avoid = True

        # 2. 检查与软管的碰撞
        for i in range(0, len(self.env.robots), 2):
            if i != self.robot_index and i+1 < len(self.env.robots):
                hose_dist = self.env.point_to_hose_min_distance(
                    cur_pos,
                    np.array([self.env.robots[i].px, self.env.robots[i].py]),
                    np.array([self.env.robots[i+1].px, self.env.robots[i+1].py])
                )
                if hose_dist < hose_safe_dist:
                    need_to_avoid = True
                    # 减速并调整方向
                    modified_direction *= 0.5

        # 3. 检查与静态human的碰撞
        for human in ob:
            if not isinstance(human, Robot):
                human_pos = np.array([human.px, human.py])
                human_dist = np.linalg.norm(human_pos - cur_pos)
                if human_dist < human_safe_dist:
                    avoid_direction = (cur_pos - human_pos) / human_dist
                    modified_direction += avoid_direction
                    need_to_avoid = True

        # 标准化方向向量
        if np.linalg.norm(modified_direction) > 0:
            modified_direction = modified_direction / np.linalg.norm(modified_direction)

        # 根据避碰情况设置速度
        speed = self.v_pref
        if need_to_avoid:
            speed *= 0.3  # 避碰时降低速度

        # 创建动作
        if self.kinematics == 'holonomic':
            action = ActionXY(modified_direction[0] * speed, modified_direction[1] * speed)
        else:
            # 非完整性机器人的运动学约束
            theta = np.arctan2(modified_direction[1], modified_direction[0])
            action = ActionRot(theta - self.theta, speed)

        return action
