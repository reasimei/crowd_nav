from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state import FullState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import numpy as np
import logging
from crowd_nav.parser import args

# robot.py

class Robot(Agent):
    def __init__(self, config, section, robot_index):
        super().__init__(config, section)
        self.robot_index = robot_index
        self.env = None  # Will be set later
        self.training_phase = 'human_avoidance'  # 添加训练阶段标识

    def set_env(self, env):
        self.env = env

    def set_training_phase(self, phase):
        self.training_phase = phase
        if hasattr(self.policy, 'set_training_phase'):
            self.policy.set_training_phase(phase)

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
        """
        生成动作，确保正确处理状态并使用策略
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        try:
            # 获取其他机器人的状态
            other_robot_states = self.get_other_robot_state()
            
            # 确保 other_robot_states 是列表
            if not isinstance(other_robot_states, list):
                other_robot_states = [other_robot_states]
            
            # 过滤无效值
            other_robot_states = [r for r in other_robot_states if r is not None]
            
            # 创建联合状态，确保训练和测试时状态设置一致
            state = JointState(self.get_full_state(), other_robot_states, ob)
            
            # 使用策略生成动作
            action = self.policy.predict(state)
            
            # 确保返回有效动作
            if action is None:
                logging.warning("Policy returned None action, using fallback")
                if self.kinematics == 'holonomic':
                    # 朝向目标的简单动作
                    direction = np.array([self.gx - self.px, self.gy - self.py])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    action = ActionXY(direction[0] * self.v_pref * 0.5, direction[1] * self.v_pref * 0.5)
                else:
                    action = ActionRot(0, 0.5)
        
            return action
        except Exception as e:
            logging.error(f"Error in act method: {e}")
            # 默认安全动作
            if self.kinematics == 'holonomic':
                direction = np.array([self.gx - self.px, self.gy - self.py])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                return ActionXY(direction[0] * self.v_pref * 0.3, direction[1] * self.v_pref * 0.3)
            else:
                return ActionRot(0, 0.3)

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
        针对训练提升的人类避碰策略
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        # 初始化默认安全动作
        if self.kinematics == 'holonomic':
            action = ActionXY(0, 0)
        else:
            action = ActionRot(0, 0)

        try:
            # 获取其他机器人状态
            other_robot_states = self.get_other_robot_state()
            if not isinstance(other_robot_states, list):
                other_robot_states = [other_robot_states]
            other_robot_states = [r for r in other_robot_states if r is not None]
            
            # 创建联合状态
            state = JointState(self.get_full_state(), other_robot_states, ob)
            
            # 获取基础动作
            base_action = self.policy.predict(state)
            if base_action is None:
                logging.warning("Policy returned None action in act_avoid_humans")
                return action  # 返回安全动作
            
            # 抽取速度
            if self.kinematics == 'holonomic':
                vx = base_action.vx
                vy = base_action.vy
            else:
                vx = base_action.v * np.cos(base_action.r + self.theta)
                vy = base_action.v * np.sin(base_action.r + self.theta)
            
            # 计算避让力 (避人类)
            force_x, force_y = 0, 0
            nearest_human_dist = float('inf')
            
            for human in ob:
                if not isinstance(human, Robot):
                    dx = self.px - human.px
                    dy = self.py - human.py
                    dist = np.sqrt(dx*dx + dy*dy)
                    nearest_human_dist = min(nearest_human_dist, dist)
                    
                    if dist < 2.0:  # 更强的避障范围
                        # 计算避障力（随距离增加）
                        force_scale = (2.0 - dist) / 2.0 * 1.5  # 更强的避障力
                        if dist > 0.001:  # 防止除以零
                            force_x += (dx/dist) * force_scale
                            force_y += (dy/dist) * force_scale
                        else:
                            # 如果距离太近，使用一个安全的默认方向
                            force_x += 1.0 * force_scale if dx >= 0 else -1.0 * force_scale
                            force_y += 1.0 * force_scale if dy >= 0 else -1.0 * force_scale
            
            # 计算避让力 (避机器人)
            nearest_robot = None
            nearest_robot_dist = float('inf')
            
            for i, robot in enumerate(self.env.robots):
                if i != self.robot_index:
                    dx = self.px - robot.px
                    dy = self.py - robot.py
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < nearest_robot_dist:
                        nearest_robot_dist = dist
                        nearest_robot = robot
                    
                    if dist < 2.5:  # 机器人避障范围更大
                        force_scale = (2.5 - dist) / 2.5 * 1.0
                        if dist > 0.001:  # 防止除以零
                            force_x += (dx/dist) * force_scale
                            force_y += (dy/dist) * force_scale
                        else:
                            # 如果距离太近，使用一个安全的默认方向
                            force_x += 1.0 * force_scale if dx >= 0 else -1.0 * force_scale
                            force_y += 1.0 * force_scale if dy >= 0 else -1.0 * force_scale
            
            # 避免机器人互相堵塞 - 添加切向分量
            if nearest_robot is not None and nearest_robot_dist < 2.5:
                dx = self.px - nearest_robot.px
                dy = self.py - nearest_robot.py
                
                # 计算切向分量（逆时针旋转90度）
                tangent_x, tangent_y = -dy, dx
                # 归一化
                tangent_len = np.sqrt(tangent_x*tangent_x + tangent_y*tangent_y)
                if tangent_len > 0.001:  # 防止除以零
                    tangent_x /= tangent_len
                    tangent_y /= tangent_len
                    
                # 基于机器人ID决定避障方向，避免对称死锁
                if self.robot_index % 2 == 0:
                    force_x += tangent_x * 0.4
                    force_y += tangent_y * 0.4
                else:
                    force_x -= tangent_x * 0.4
                    force_y -= tangent_y * 0.4
            
            # 软管约束（如果有）
            if hasattr(self.env, 'hose_length') and self.env.hose_length > 0:
                try:
                    partner_idx = self.robot_index + 1 if self.robot_index % 2 == 0 else self.robot_index - 1
                    if 0 <= partner_idx < len(self.env.robots):
                        hose_partner = self.env.robots[partner_idx]
                        dx = self.px - hose_partner.px
                        dy = self.py - hose_partner.py
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        # 太近时排斥
                        if dist < 1.0:
                            if dist > 0.001:  # 防止除以零
                                force_x += (dx/dist) * 1.2
                                force_y += (dy/dist) * 1.2
                            else:
                                # 随机方向避开
                                force_x += (np.random.random() - 0.5) * 1.2
                                force_y += (np.random.random() - 0.5) * 1.2
                        # 太远时吸引
                        elif dist > self.env.hose_length * 0.8:
                            if dist > 0.001:  # 防止除以零
                                force_x -= (dx/dist) * 0.8
                                force_y -= (dy/dist) * 0.8
                            else:
                                # 随机方向避开（不太可能发生但仍要处理）
                                force_x -= (np.random.random() - 0.5) * 0.8
                                force_y -= (np.random.random() - 0.5) * 0.8
                except Exception as e:
                    logging.debug(f"Hose constraint error: {e}")
            
            # 组合力和基础速度
            final_vx = vx + force_x
            final_vy = vy + force_y
            
            # 速度限制
            speed = np.sqrt(final_vx*final_vx + final_vy*final_vy)
            if speed > self.v_pref:
                final_vx = final_vx * self.v_pref / speed
                final_vy = final_vy * self.v_pref / speed
            
            # 创建最终动作
            if self.kinematics == 'holonomic':
                action = ActionXY(final_vx, final_vy)
            else:
                theta = np.arctan2(final_vy, final_vx)
                action = ActionRot(theta - self.theta, speed)
            
        except Exception as e:
            logging.error(f"Error in act_avoid_humans: {e}")
            # 使用默认动作
        
        return action

    def act_avoid_robots(self, ob, flag):
        """
        机器人避让策略,考虑:
        1. 与其他机器人的避碰
        2. 与软管的避碰
        3. 与静态human的避碰
        4. 根据位置设置不同运动策略
        """
        # 获取当前位置和目标位置
        cur_pos = np.array([self.px, self.py])
        goal_pos = np.array([self.gx, self.gy])
        
        # 初始化速度和方向
        preferred_speed = self.v_pref
        speed = preferred_speed  # 初始化速度
        direction = goal_pos - cur_pos
        dist_to_goal = np.linalg.norm(direction)
        
        if dist_to_goal > 0:
            direction = direction / dist_to_goal
        else:
            return ActionXY(0, 0)
        
        modified_direction = direction.copy()
        need_to_avoid = False
        
        # 1. 设置安全距离
        robot_safe_dist = 2.0
        human_safe_dist = 1.0
        hose_safe_dist = 1.5   # 与软管安全距离
        
        # 2. 检查与其他机器人的碰撞
        for other_robot in ob:
            if isinstance(other_robot, Robot) and other_robot != self:
                other_pos = np.array([other_robot.px, other_robot.py])
                robot_dist = np.linalg.norm(other_pos - cur_pos)
                if robot_dist < robot_safe_dist:
                    avoid_direction = (cur_pos - other_pos) / robot_dist
                    modified_direction += avoid_direction
                    need_to_avoid = True
                    speed *= 0.5  # 降低速度
        
        # 3. 检查与软管的碰撞
        # 安全地检查是否使用软管
        use_hose = False
        try:
            from crowd_nav.parser import args
            use_hose = args.hose
        except:
            # 如果导入失败，保持默认值
            pass
        
        if use_hose:
            try:
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
                            speed *= 0.5
            except Exception as e:
                logging.debug(f"Error in hose collision check: {e}")
        
        # 4. 检查与静态human的碰撞
        for human in ob:
            if not isinstance(human, Robot):
                human_pos = np.array([human.px, human.py])
                human_dist = np.linalg.norm(human_pos - cur_pos)
                # 计算避障方向
                avoid_direction = np.array([0, 0])
                if human_dist > 0.001:  # 防止除以零
                    avoid_direction = (cur_pos - human_pos) / human_dist
                else:
                    # 如果距离太近，使用一个安全的默认方向
                    avoid_direction = np.array([1.0, 0.0])  # 默认向右避开
                
                if hasattr(self, 'training_phase') and self.training_phase == 'robot_avoidance':
                    # 在 robot_avoidance 阶段，采用绕行策略
                    if human_dist < human_safe_dist:
                        # 计算切向方向（垂直于避障方向）
                        tangent = np.array([-avoid_direction[1], avoid_direction[0]])
                        # 根据机器人在human左边还是右边决定绕行方向
                        if np.dot(tangent, (goal_pos - cur_pos)) > 0:
                            modified_direction += tangent
                        else:
                            modified_direction -= tangent
                        need_to_avoid = True
                        speed *= 0.7  # 绕行时适度降低速度
                else:
                    # 在 human_avoidance 阶段，保持原有的保守避障策略
                    if human_dist < human_safe_dist:
                        modified_direction += avoid_direction
                        need_to_avoid = True
                        speed *= 0.3  # 保守避障时大幅降低速度
        
        # 5. 标准化方向向量
        if need_to_avoid:
            norm = np.linalg.norm(modified_direction)
            if norm > 0:
                modified_direction = modified_direction / norm
        
        # 6. 计算最终速度分量
        vx = speed * modified_direction[0]
        vy = speed * modified_direction[1]
        
        return ActionXY(vx, vy)
