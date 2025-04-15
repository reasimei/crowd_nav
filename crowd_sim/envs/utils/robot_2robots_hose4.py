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
        Modified to handle both human avoidance and hose constraints
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        # Safety parameters
        safe_dist_humans = 2.0
        safe_dist_hose_partner = 1.5  # Minimum distance to maintain with connected robot
        min_hose_dist = 1.0  # Critical minimum distance
        optimal_hose_dist = self.env.hose_length * 0.7  # Target distance with partner

        # Get hose partner (robots are paired sequentially)
        partner_idx = self.robot_index + 1 if self.robot_index % 2 == 0 else self.robot_index - 1
        if 0 <= partner_idx < len(self.env.robots):
            hose_partner = self.env.robots[partner_idx]
            dist_to_partner = np.linalg.norm(np.array([self.px - hose_partner.px, 
                                                    self.py - hose_partner.py]))
        else:
            hose_partner = None
            dist_to_partner = float('inf')

        
        # Get state for policy
        state = JointState(self.get_full_state(), [], ob)
        #state = JointState(self.get_other_robot_state(), self.get_full_state())
        
        # Get base action from policy
        base_action = self.policy.predict(state)
        
        # Convert action to velocity components
        if self.kinematics == 'holonomic':
            vx = base_action.vx
            vy = base_action.vy
        else:
            vx = base_action.v * np.cos(base_action.r + self.theta)
            vy = base_action.v * np.sin(base_action.r + self.theta)

        # Calculate repulsive forces from humans
        force_x, force_y = 0, 0
        for human in ob:
            if not isinstance(human, Robot):
                dx = self.px - human.px
                dy = self.py - human.py
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < safe_dist_humans:
                    force_x += (dx/dist) * (safe_dist_humans - dist) / safe_dist_humans
                    force_y += (dy/dist) * (safe_dist_humans - dist) / safe_dist_humans

        # Add hose partner avoidance
        if hose_partner:
            dx = self.px - hose_partner.px
            dy = self.py - hose_partner.py
            
            # Strong repulsion if too close
            if dist_to_partner < min_hose_dist:
                force_scale = 2.0 * (min_hose_dist - dist_to_partner) / min_hose_dist
                force_x += (dx/dist_to_partner) * force_scale
                force_y += (dy/dist_to_partner) * force_scale
            
            # Attraction if too far
            elif dist_to_partner > self.env.hose_length:
                force_scale = 0.5 * (dist_to_partner - self.env.hose_length) / self.env.hose_length
                force_x -= (dx/dist_to_partner) * force_scale
                force_y -= (dy/dist_to_partner) * force_scale

        # Combine base velocity with forces
        final_vx = vx + force_x
        final_vy = vy + force_y
        
        # Scale velocity based on distance to partner
        if hose_partner:
            if dist_to_partner < min_hose_dist:
                speed_scale = 0.2  # Severe speed reduction
            elif dist_to_partner < safe_dist_hose_partner:
                speed_scale = 0.5  # Moderate speed reduction
            else:
                speed_scale = 1.0
                
            final_vx *= speed_scale
            final_vy *= speed_scale

        # Create final action
        if self.kinematics == 'holonomic':
            action = ActionXY(final_vx, final_vy)
        else:
            # Convert to angular motion
            theta = np.arctan2(final_vy, final_vx)
            speed = np.sqrt(final_vx**2 + final_vy**2)
            action = ActionRot(theta - self.theta, speed)

        return action

        # except Exception as e:
        #     logging.error(f"Error in act_avoid_humans: {e}")
        #     return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)


    def act_avoid_robots(self, ob):
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
        
        # 4. 检查与静态human的碰撞
        for human in ob:
            if not isinstance(human, Robot):
                human_pos = np.array([human.px, human.py])
                human_dist = np.linalg.norm(human_pos - cur_pos)
                # 计算避障方向
                avoid_direction = (cur_pos - human_pos) / human_dist if human_dist > 0 else np.array([0, 0])
                
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
