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
        self.hose_partner = None  # 软管伙伴引用
        self.last_hose_points = None  # 上一次计算的软管形状
        self.hose_safe_distance = 0.6  # 软管安全距离
        self.last_human_hose_distances = {}  # 跟踪人类与软管的距离

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
        针对训练提升的人类避碰策略，增强软管安全性
        """
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        # 初始化默认安全动作
        if self.kinematics == 'holonomic':
            action = ActionXY(0, 0)
        else:
            action = ActionRot(0, 0)

        try:
            # 检查是否所有人类已到达目标
            all_humans_at_goal = False
            humans_stable_at_goal = False
            
            if hasattr(self.env, 'all_humans_at_goal'):
                all_humans_at_goal = self.env.all_humans_at_goal
                # 检查人类是否稳定在目标位置
                if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                    humans_stable_at_goal = all_humans_at_goal and (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold
            
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
            
            # 如果人类稳定在目标点且机器人离目标很近，快速直接移动到目标
            if humans_stable_at_goal:
                # 计算到目标的距离
                dist_to_goal = np.linalg.norm([self.px - self.gx, self.py - self.gy])
                
                # 距离目标很近时，加快直接移动到目标
                if dist_to_goal < 1.0:
                    goal_dir = np.array([self.gx - self.px, self.gy - self.py])
                    if np.linalg.norm(goal_dir) > 0:
                        goal_dir = goal_dir / np.linalg.norm(goal_dir)
                        # 直接向目标移动，速度与距离成正比
                        speed = min(self.v_pref, max(0.3, dist_to_goal * 0.8))
                        return ActionXY(goal_dir[0] * speed, goal_dir[1] * speed)
            
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
            
            # 提取人类观测
            human_states = []
            for entity in ob:
                if not isinstance(entity, Robot):
                    human_states.append(entity)
            
            # 当人类静止时减少避障力
            human_force_factor = 0.5 if humans_stable_at_goal else 1.0
            
            for human in human_states:
                dx = self.px - human.px
                dy = self.py - human.py
                dist = np.sqrt(dx*dx + dy*dy)
                nearest_human_dist = min(nearest_human_dist, dist)
                
                # 如果人类静止在目标点，缩小避障范围
                avoid_range = 1.0 if humans_stable_at_goal else 2.0
                
                if dist < avoid_range:  
                    # 计算避障力（随距离增加）
                    force_scale = (avoid_range - dist) / avoid_range * 1.5 * human_force_factor
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
            
            # 当人类静止时减少机器人间避障力
            robot_force_factor = 0.6 if humans_stable_at_goal else 1.0
            
            for i, robot in enumerate(self.env.robots):
                if i != self.robot_index:
                    dx = self.px - robot.px
                    dy = self.py - robot.py
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < nearest_robot_dist:
                        nearest_robot_dist = dist
                        nearest_robot = robot
                    
                    # 如果人类静止，缩小机器人避障范围
                    robot_avoid_range = 1.5 if humans_stable_at_goal else 2.5
                    
                    if dist < robot_avoid_range:
                        force_scale = (robot_avoid_range - dist) / robot_avoid_range * 1.0 * robot_force_factor
                        if dist > 0.001:  # 防止除以零
                            force_x += (dx/dist) * force_scale
                            force_y += (dy/dist) * force_scale
                        else:
                            # 如果距离太近，使用一个安全的默认方向
                            force_x += 1.0 * force_scale if dx >= 0 else -1.0 * force_scale
                            force_y += 1.0 * force_scale if dy >= 0 else -1.0 * force_scale
            
            # 软管安全力 - 使用新增的计算方法
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            if use_hose:
                # 计算软管安全力和危险级别
                hose_force_x, hose_force_y, hose_danger = self._calculate_hose_safety_force(human_states)
                
                # 增加软管避障的力度
                force_x += hose_force_x
                force_y += hose_force_y
                
                # 伙伴机器人距离管理
                partner = self.get_hose_partner()
                if partner is not None:
                    dx = self.px - partner.px
                    dy = self.py - partner.py
                    dist = np.sqrt(dx*dx + dy*dy)
                    hose_length = getattr(self.env, 'hose_length', 0)
                    
                    # 协调动作因子 - 人类静止时增强协调性
                    coordination_factor = 1.2 if humans_stable_at_goal else 0.8
                    
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
                    elif hose_length > 0 and dist > hose_length * 0.8:
                        if dist > 0.001:  # 防止除以零
                            force_x -= (dx/dist) * 0.8 * coordination_factor
                            force_y -= (dy/dist) * 0.8 * coordination_factor
                        else:
                            # 随机方向避开（不太可能发生但仍要处理）
                            force_x -= (np.random.random() - 0.5) * 0.8 * coordination_factor
                            force_y -= (np.random.random() - 0.5) * 0.8 * coordination_factor
                    
                    # 碰撞风险高时降低速度
                    if hose_danger and dist > hose_length * 0.5:
                        # 尽量减少软管张力，靠近伙伴
                        pull_scale = 0.8 * coordination_factor
                        if dist > 0.001:
                            force_x -= (dx/dist) * pull_scale
                            force_y -= (dy/dist) * pull_scale
                    
                    # 机器人伙伴协调 - 获取伙伴的速度并尝试协调
                    if hasattr(partner, 'policy') and hasattr(partner.policy, 'last_velocity'):
                        # 获取伙伴的当前速度向量
                        partner_vx, partner_vy = 0, 0
                        if hasattr(partner.policy, 'last_velocity'):
                            partner_vx, partner_vy = partner.policy.last_velocity
                        
                        # 当人类静止时，更强地协调彼此的速度
                        if humans_stable_at_goal:
                            # 计算当前机器人到目标的方向
                            my_goal_dir = np.array([self.gx - self.px, self.gy - self.py])
                            my_goal_dist = np.linalg.norm(my_goal_dir)
                            if my_goal_dist > 0:
                                my_goal_dir = my_goal_dir / my_goal_dist
                            
                            # 计算伙伴到目标的方向
                            partner_goal_dir = np.array([partner.gx - partner.px, partner.gy - partner.py])
                            partner_goal_dist = np.linalg.norm(partner_goal_dir)
                            if partner_goal_dist > 0:
                                partner_goal_dir = partner_goal_dir / partner_goal_dist
                            
                            # 计算两个目标方向的点积，判断是否朝相似方向
                            direction_alignment = np.dot(my_goal_dir, partner_goal_dir)
                            
                            # 如果伙伴和我的目标方向大致一致，加强协调
                            if direction_alignment > 0.5:
                                # 协调系数随方向一致性增加
                                coordination_strength = 0.3 * (1 + direction_alignment)
                                
                                # 调整我的速度朝向伙伴的速度方向
                                vx += partner_vx * coordination_strength
                                vy += partner_vy * coordination_strength
                                
                                # 同时调整力向量减弱与伙伴动作的冲突
                                partner_force = np.array([partner_vx, partner_vy])
                                my_force = np.array([force_x, force_y])
                                
                                # 如果力向量与伙伴速度方向相反，减弱它
                                if np.dot(partner_force, my_force) < 0:
                                    force_factor = 0.5  # 减弱相反的力
                                    force_x *= force_factor
                                    force_y *= force_factor
            
            # 组合力和基础速度
            final_vx = vx + force_x
            final_vy = vy + force_y
            
            # 根据软管碰撞风险调整速度
            if use_hose and 'hose_danger' in locals() and hose_danger:
                # 有危险时大幅降低速度，但人类静止时减轻程度
                speed_scale = 0.6 if humans_stable_at_goal else 0.4
                final_vx *= speed_scale
                final_vy *= speed_scale
                logging.debug("降低速度以避免软管碰撞")
            
            # 人类静止在目标点时，确保有足够的速度朝向目标
            if humans_stable_at_goal:
                # 计算到目标的向量
                goal_vec = np.array([self.gx - self.px, self.gy - self.py])
                goal_dist = np.linalg.norm(goal_vec)
                
                if goal_dist > 0.3:  # 离目标还有一定距离
                    # 归一化目标向量
                    if goal_dist > 0:
                        goal_vec = goal_vec / goal_dist
                    
                    # 计算当前速度向目标方向的投影
                    current_speed = np.array([final_vx, final_vy])
                    speed_magnitude = np.linalg.norm(current_speed)
                    
                    if speed_magnitude > 0:
                        goal_alignment = np.dot(current_speed / speed_magnitude, goal_vec)
                        
                        # 如果当前速度不够或方向不佳，增加朝向目标的分量
                        if speed_magnitude < 0.3 or goal_alignment < 0.5:
                            # 添加朝向目标的速度分量
                            goal_boost = 0.3 * goal_vec
                            final_vx += goal_boost[0]
                            final_vy += goal_boost[1]
            
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
        
        # 检查是否所有人类已到达目标
        all_humans_at_goal = False
        humans_stable_at_goal = False
        
        if hasattr(self.env, 'all_humans_at_goal'):
            all_humans_at_goal = self.env.all_humans_at_goal
            # 检查人类是否稳定在目标位置
            if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                humans_stable_at_goal = all_humans_at_goal and (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold
        
        # 提取人类观测
        human_states = []
        for entity in ob:
            if not isinstance(entity, Robot):
                human_states.append(entity)
        
        # 1. 设置安全距离
        robot_safe_dist = 1.5 if humans_stable_at_goal else 2.0
        human_safe_dist = 0.8 if humans_stable_at_goal else 1.0
        hose_safe_dist = 1.2 if humans_stable_at_goal else 1.5   # 与软管安全距离
        
        # 2. 检查与其他机器人的碰撞
        for other_robot in ob:
            if isinstance(other_robot, Robot) and other_robot != self:
                other_pos = np.array([other_robot.px, other_robot.py])
                robot_dist = np.linalg.norm(other_pos - cur_pos)
                if robot_dist < robot_safe_dist:
                    avoid_direction = (cur_pos - other_pos) / robot_dist
                    
                    # 当人类静止时，根据与目标的关系调整避让强度
                    if humans_stable_at_goal:
                        # 计算避让方向与目标方向的点积
                        goal_alignment = np.dot(avoid_direction, direction)
                        
                        # 如果避让方向与目标方向一致，增强避让
                        if goal_alignment > 0:
                            modified_direction += avoid_direction * 1.5
                        else:
                            # 如果避让方向与目标方向冲突，适度降低避让强度
                            modified_direction += avoid_direction * 0.8
                    else:
                        modified_direction += avoid_direction
                    
                    need_to_avoid = True
                    speed *= 0.5  # 降低速度
        
        # 3. 检查与软管的碰撞
        # 安全地检查是否使用软管
        use_hose = False
        try:
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
        except:
            # 如果导入失败，保持默认值
            pass
        
        if use_hose:
            try:
                # 获取当前机器人的软管对编号
                curr_pair_idx = self.robot_index // 2
                
                # 记录最近的其他软管信息
                closest_hose_dist = float('inf')
                closest_hose_avoid_dir = None
                
                # 检查所有其他软管对
                for i in range(0, len(self.env.robots), 2):
                    pair_idx = i // 2
                    # 跳过自己的软管对
                    if pair_idx == curr_pair_idx or i+1 >= len(self.env.robots):
                        continue
                    
                    # 获取其他软管对的两个机器人
                    other_robot1 = self.env.robots[i]
                    other_robot2 = self.env.robots[i+1]
                    
                    # 使用点到软管的最短距离函数
                    hose_dist = self.env.point_to_hose_min_distance(
                        cur_pos,
                        np.array([other_robot1.px, other_robot1.py]),
                        np.array([other_robot2.px, other_robot2.py])
                    )
                    
                    if hose_dist < hose_safe_dist:
                        # 计算其他软管的中点
                        other_hose_midpoint = np.array([
                            (other_robot1.px + other_robot2.px) / 2,
                            (other_robot1.py + other_robot2.py) / 2
                        ])
                        
                        # 计算从其他软管中点到当前位置的向量
                        avoid_direction = cur_pos - other_hose_midpoint
                        
                        # 归一化
                        avoid_dist = np.linalg.norm(avoid_direction)
                        if avoid_dist > 0.001:
                            avoid_direction = avoid_direction / avoid_dist
                            
                            # 判断是否为最近的软管
                            if hose_dist < closest_hose_dist:
                                closest_hose_dist = hose_dist
                                closest_hose_avoid_dir = avoid_direction
                            
                            # 计算软管方向
                            hose_dir = np.array([
                                other_robot2.px - other_robot1.px,
                                other_robot2.py - other_robot1.py
                            ])
                            if np.linalg.norm(hose_dir) > 0.001:
                                hose_dir = hose_dir / np.linalg.norm(hose_dir)
                                
                                # 计算垂直方向，可能提供更好的避让
                                perp1 = np.array([-hose_dir[1], hose_dir[0]])
                                perp2 = -perp1
                                
                                # 选择更接近目标的垂直方向
                                if np.dot(perp1, direction) > np.dot(perp2, direction):
                                    perp = perp1
                                else:
                                    perp = perp2
                                
                                # 当人类静止时，增加垂直避让的比例
                                if humans_stable_at_goal:
                                    # 更多地使用垂直方向避让，以寻找更好的绕行路径
                                    avoid_scale = (hose_safe_dist - hose_dist) / hose_safe_dist
                                    modified_direction += perp * avoid_scale * 0.8
                                    modified_direction += avoid_direction * avoid_scale * 0.4
                                else:
                                    avoid_scale = (hose_safe_dist - hose_dist) / hose_safe_dist
                                    modified_direction += avoid_direction * avoid_scale
                            else:
                                # 如果无法计算软管方向，使用直接避让
                                avoid_scale = (hose_safe_dist - hose_dist) / hose_safe_dist
                                modified_direction += avoid_direction * avoid_scale
                            
                            need_to_avoid = True
                            # 根据距离调整速度，越近速度越低
                            speed_factor = max(0.3, hose_dist / hose_safe_dist)
                            speed *= speed_factor
                
                # 如果检测到多个软管，且人类已静止，尝试找到更优的全局路径
                if humans_stable_at_goal and closest_hose_avoid_dir is not None:
                    # 距离目标较远时，可考虑对路径进行更大的调整
                    if dist_to_goal > 2.0:
                        # 计算朝向目标的方向与避让方向的关系
                        goal_alignment = np.dot(closest_hose_avoid_dir, direction)
                        
                        if goal_alignment < -0.5:  # 方向高度冲突
                            # 为了避免死锁，根据机器人ID选择不同的避让策略
                            if self.robot_index % 4 < 2:
                                # 尝试通过更大的绕行避让
                                # 使用垂直于软管的方向
                                perp_dir = np.array([-closest_hose_avoid_dir[1], closest_hose_avoid_dir[0]])
                                modified_direction = direction * 0.3 + perp_dir * 0.7
                            else:
                                # 使用相反的垂直方向，确保不同对的机器人选择不同路径
                                perp_dir = np.array([closest_hose_avoid_dir[1], -closest_hose_avoid_dir[0]])
                                modified_direction = direction * 0.3 + perp_dir * 0.7
                        
                        # 如果接近目标，尝试更直接的路径
                        if dist_to_goal < 1.5 and goal_alignment < 0:
                            # 更偏向于目标方向，以确保能够到达
                            modified_direction = direction * 0.7 + closest_hose_avoid_dir * 0.3
                
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
                        avoid_scale = (human_safe_dist - human_dist) / human_safe_dist
                        modified_direction += avoid_direction * avoid_scale
                        need_to_avoid = True
                        # 人类静止时适度提高速度
                        speed *= 0.5 if not humans_stable_at_goal else 0.7
        
        # 5. 标准化方向向量
        if need_to_avoid:
            norm = np.linalg.norm(modified_direction)
            if norm > 0:
                modified_direction = modified_direction / norm
        
        # 6. 当人类静止且接近目标时，增强目标导向
        if humans_stable_at_goal and dist_to_goal < 1.0:
            # 增加朝目标方向的权重
            modified_direction = modified_direction * 0.3 + direction * 0.7
            
            # 重新标准化
            norm = np.linalg.norm(modified_direction)
            if norm > 0:
                modified_direction = modified_direction / norm
            
            # 根据距离适度增加速度
            speed = preferred_speed * min(1.0, max(0.5, dist_to_goal * 0.8))
        
        # 7. 计算最终速度分量
        vx = speed * modified_direction[0]
        vy = speed * modified_direction[1]
        
        return ActionXY(vx, vy)

    def get_hose_partner(self):
        """
        获取当前机器人的软管伙伴
        
        Returns:
            Robot: 与当前机器人通过软管连接的伙伴，如果没有则返回None
        """
        try:
            # 检查是否使用软管
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            if not use_hose or self.env is None:
                return None
            
            # 找到伙伴索引 (机器人是两两配对: 0-1, 2-3, 4-5...)
            partner_idx = self.robot_index + 1 if self.robot_index % 2 == 0 else self.robot_index - 1
            
            # 确保伙伴索引有效
            if 0 <= partner_idx < len(self.env.robots):
                if self.hose_partner is None:  # 缓存伙伴引用
                    self.hose_partner = self.env.robots[partner_idx]
                return self.hose_partner
            return None
        except Exception as e:
            logging.debug(f"Error getting hose partner: {e}")
            return None
    
    def estimate_hose_curve(self):
        """
        估计当前机器人与其伙伴之间的软管形状
        
        Returns:
            tuple: (x_points, y_points) 表示软管形状的点集，如果没有软管则返回None
        """
        try:
            partner = self.get_hose_partner()
            if partner is None:
                return None
            
            # 检查是否有软管长度参数
            hose_length = getattr(self.env, 'hose_length', 0)
            if hose_length <= 0:
                return None
            
            # 导入工具函数
            from crowd_sim.envs.utils.utils import hose_model
            
            # 计算软管形状
            robot1_pos = (self.px, self.py)
            robot2_pos = (partner.px, partner.py)
            
            # 获取软管的形状点集
            x_points, y_points = hose_model(robot1_pos, robot2_pos, hose_length)
            
            # 缓存结果供后续使用
            self.last_hose_points = (x_points, y_points)
            
            return (x_points, y_points)
        except Exception as e:
            logging.debug(f"Error estimating hose curve: {e}")
            return self.last_hose_points  # 出错时返回上次计算的结果
    
    def predict_hose_collision(self, human_states):
        """
        预测软管是否会与人类和其他机器人软管发生碰撞
        
        Args:
            human_states: 人类状态列表
            
        Returns:
            tuple: (collision_risk, closest_entity, min_distance, is_robot_hose)
                - collision_risk: 碰撞风险值 (0-1)
                - closest_entity: 最近的实体（人类或机器人）
                - min_distance: 最短距离
                - is_robot_hose: 是否与其他机器人软管发生碰撞
        """
        try:
            # 检查是否使用软管
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            if not use_hose:
                return 0, None, float('inf'), False
            
            # 获取软管伙伴
            partner = self.get_hose_partner()
            if partner is None:
                return 0, None, float('inf'), False
                
            # 获取软管长度
            hose_length = getattr(self.env, 'hose_length', 0)
            if hose_length <= 0:
                return 0, None, float('inf'), False
                
            # 从环境中获取point_to_hose_curve函数
            from crowd_sim.envs.utils.utils import point_to_hose_curve
            
            # 检查每个人与软管的最短距离
            min_distance = float('inf')
            closest_entity = None
            is_robot_hose = False
            
            # 先检查与人类的碰撞
            for human in human_states:
                # 计算人到软管的最短距离
                human_pos = (human.px, human.py)
                distance = point_to_hose_curve(
                    human_pos, 
                    (self.px, self.py), 
                    (partner.px, partner.py), 
                    hose_length
                ) - human.radius - self.env.hose_thickness
                
                # 更新距离跟踪
                self.last_human_hose_distances[human] = distance
                
                # 找到最小距离
                if distance < min_distance:
                    min_distance = distance
                    closest_entity = human
                    is_robot_hose = False
            
            # 再检查与其他机器人软管的碰撞
            # 只检查与当前机器人-软管对不同的软管
            curr_pair_idx = self.robot_index // 2  # 当前机器人软管对的索引
            
            for i in range(0, len(self.env.robots), 2):
                pair_idx = i // 2
                # 跳过自己的软管对
                if pair_idx == curr_pair_idx or i+1 >= len(self.env.robots):
                    continue
                
                # 获取其他软管对的两个机器人
                other_robot1 = self.env.robots[i]
                other_robot2 = self.env.robots[i+1]
                
                # 检查当前机器人与其他软管的碰撞
                # 测试多个点而不仅仅是机器人中心
                test_points = []
                # 添加机器人自身位置
                test_points.append((self.px, self.py, self.radius))
                
                # 添加沿着自身软管的几个点
                if partner is not None:
                    hose_vec = np.array([partner.px - self.px, partner.py - self.py])
                    hose_len = np.linalg.norm(hose_vec)
                    if hose_len > 0.001:
                        hose_dir = hose_vec / hose_len
                        # 在软管上添加几个测试点
                        for t in [0.25, 0.5, 0.75]:
                            pos_x = self.px + hose_dir[0] * hose_len * t
                            pos_y = self.py + hose_dir[1] * hose_len * t
                            test_points.append((pos_x, pos_y, self.env.hose_thickness))
                
                # 检查每个测试点到其他软管的距离
                for pos_x, pos_y, radius in test_points:
                    test_pos = (pos_x, pos_y)
                    distance = point_to_hose_curve(
                        test_pos,
                        (other_robot1.px, other_robot1.py),
                        (other_robot2.px, other_robot2.py),
                        hose_length
                    ) - radius - self.env.hose_thickness
                    
                    if distance < min_distance:
                        min_distance = distance
                        # 将两个机器人作为一个元组存储，表示与哪个软管发生碰撞
                        closest_entity = (other_robot1, other_robot2)
                        is_robot_hose = True
            
            # 计算碰撞风险 (距离越近风险越高)
            collision_risk = 0
            if min_distance < 2.0:
                # 如果是与其他软管的碰撞，增加风险权重
                if is_robot_hose:
                    collision_risk = max(0, 1.2 - min_distance / 2.0)  # 提高对其他软管的避让优先级
                else:
                    collision_risk = max(0, 1.0 - min_distance / 2.0)
                
                # 防止值过大
                collision_risk = min(collision_risk, 1.0)
            
            return collision_risk, closest_entity, min_distance, is_robot_hose
        except Exception as e:
            logging.debug(f"Error predicting hose collision: {e}")
            return 0, None, float('inf'), False
    
    def _min_distance_between_segments(self, p1, p2, p3, p4):
        """
        计算两条线段之间的最短距离
        
        Args:
            p1, p2: 第一条线段的端点
            p3, p4: 第二条线段的端点
            
        Returns:
            float: 两条线段之间的最短距离
        """
        def dot(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]
            
        def distance_to_point(p, s1, s2):
            # 计算点p到线段s1-s2的最短距离
            v = s2 - s1
            w = p - s1
            c1 = dot(w, v)
            if c1 <= 0:
                return np.linalg.norm(p - s1)
            c2 = dot(v, v)
            if c2 <= c1:
                return np.linalg.norm(p - s2)
            b = c1 / c2
            pb = s1 + b * v
            return np.linalg.norm(p - pb)
        
        # 检查两条线段是否相交
        v1 = p2 - p1
        v2 = p4 - p3
        cross_v1v2 = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(cross_v1v2) < 1e-10:  # 平行或共线
            # 计算点到线段的距离
            d1 = distance_to_point(p3, p1, p2)
            d2 = distance_to_point(p4, p1, p2)
            d3 = distance_to_point(p1, p3, p4)
            d4 = distance_to_point(p2, p3, p4)
            return min(d1, d2, d3, d4)
        
        # 计算交点
        t1 = ((p3[0] - p1[0]) * v2[1] - (p3[1] - p1[1]) * v2[0]) / cross_v1v2
        t2 = ((p1[0] - p3[0]) * v1[1] - (p1[1] - p3[1]) * v1[0]) / -cross_v1v2
        
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            # 两条线段相交
            return 0.0
        
        # 计算端点到线段的距离
        d1 = distance_to_point(p3, p1, p2)
        d2 = distance_to_point(p4, p1, p2)
        d3 = distance_to_point(p1, p3, p4)
        d4 = distance_to_point(p2, p3, p4)
        
        return min(d1, d2, d3, d4)

    def _calculate_hose_safety_force(self, human_states):
        """
        计算保持软管安全的力向量
        
        Args:
            human_states: 人类状态列表
            
        Returns:
            tuple: (force_x, force_y, hose_danger) 软管安全力向量和危险级别
        """
        force_x, force_y = 0, 0
        hose_danger = False
        
        try:
            # 检查是否使用软管
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            if not use_hose:
                return force_x, force_y, hose_danger
            
            # 检查是否所有人类已到达目标
            all_humans_at_goal = False
            humans_stable_at_goal = False
            
            if hasattr(self.env, 'all_humans_at_goal'):
                all_humans_at_goal = self.env.all_humans_at_goal
                # 检查人类是否稳定在目标位置
                if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                    humans_stable_at_goal = all_humans_at_goal and (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold
            
            # 获取碰撞风险信息
            collision_risk, closest_entity, min_distance, is_robot_hose = self.predict_hose_collision(human_states)
            
            # 如果有碰撞风险
            if collision_risk > 0:
                partner = self.get_hose_partner()
                if partner is None or closest_entity is None:
                    return force_x, force_y, hose_danger
                
                # 计算软管中点
                hose_midpoint = np.array([
                    (self.px + partner.px) / 2, 
                    (self.py + partner.py) / 2
                ])
                
                # 计算避让向量
                if is_robot_hose:
                    # 处理与其他机器人软管的避让
                    other_robot1, other_robot2 = closest_entity
                    
                    # 计算其他软管的中点
                    other_hose_midpoint = np.array([
                        (other_robot1.px + other_robot2.px) / 2,
                        (other_robot1.py + other_robot2.py) / 2
                    ])
                    
                    # 计算从其他软管中点到当前软管中点的向量
                    avoid_vector = hose_midpoint - other_hose_midpoint
                    
                    # 归一化
                    dist = np.linalg.norm(avoid_vector)
                    if dist > 0.001:
                        avoid_vector = avoid_vector / dist
                    else:
                        # 如果太近，使用从其他软管方向垂直的方向
                        other_hose_dir = np.array([
                            other_robot2.px - other_robot1.px,
                            other_robot2.py - other_robot1.py
                        ])
                        if np.linalg.norm(other_hose_dir) > 0.001:
                            other_hose_dir = other_hose_dir / np.linalg.norm(other_hose_dir)
                            # 垂直方向
                            avoid_vector = np.array([-other_hose_dir[1], other_hose_dir[0]])
                            
                            # 根据机器人ID决定避让方向，避免对称死锁
                            if self.robot_index % 4 >= 2:  # 不同软管对选择不同方向
                                avoid_vector = -avoid_vector
                else:
                    # 处理与人类的避让
                    # 计算人类到软管中点的向量
                    avoid_vector = hose_midpoint - np.array([closest_entity.px, closest_entity.py])
                    
                    # 归一化
                    dist = np.linalg.norm(avoid_vector)
                    if dist > 0.001:
                        avoid_vector = avoid_vector / dist
                
                # 计算软管方向
                hose_direction = np.array([partner.px - self.px, partner.py - self.py])
                hose_length = np.linalg.norm(hose_direction)
                
                if hose_length > 0.001:
                    hose_direction = hose_direction / hose_length
                    
                    # 计算垂直于软管的方向 (两个方向，取与避让方向最接近的一个)
                    perp1 = np.array([-hose_direction[1], hose_direction[0]])
                    perp2 = -perp1
                    
                    # 选择与避让方向最接近的垂直方向
                    if np.dot(perp1, avoid_vector) > np.dot(perp2, avoid_vector):
                        perp = perp1
                    else:
                        perp = perp2
                    
                    # 根据风险程度计算力度
                    # 当人类静止时调整安全力度
                    force_magnitude = collision_risk * 1.5
                    
                    if is_robot_hose:
                        # 对于机器人软管，即使人类静止也保持较高权重
                        if all_humans_at_goal:
                            # 人类静止但仍与其他软管保持较高避让优先级
                            force_magnitude *= 0.8  # 较小的减少
                        
                        # 如果人类已经稳定在目标点，根据碰撞风险调整力度
                        if humans_stable_at_goal:
                            if collision_risk > 0.7:  # 高风险
                                force_magnitude *= 0.9  # 几乎不减少
                            else:  # 低风险
                                force_magnitude *= 0.7  # 适度减少
                    else:
                        # 对于人类，当人类静止时大幅度降低避让优先级
                        if all_humans_at_goal:
                            # 人类静止时减少力度
                            force_magnitude *= 0.5
                        
                        # 如果人类已经稳定在目标点且碰撞风险较低，进一步降低安全力
                        if humans_stable_at_goal and collision_risk < 0.5:
                            force_magnitude *= 0.3
                    
                    # 如果距离非常近，标记为危险
                    if min_distance < 0.3:
                        hose_danger = True
                        # 根据是否为机器人软管调整危险时的力度
                        if is_robot_hose:
                            # 对于机器人软管保持更高的避让力度
                            force_magnitude = max(force_magnitude, 1.2)
                        else:
                            # 对于人类维持原有逻辑
                            force_magnitude = max(force_magnitude, 1.0)
                    
                    # 计算最终力向量 (垂直方向为主，微调朝向避让方向)
                    # 主要使用垂直方向，但加入少量直接避让方向
                    if is_robot_hose:
                        # 对于机器人软管，使用更多直接避让方向的成分
                        force_x = perp[0] * force_magnitude * 0.7 + avoid_vector[0] * force_magnitude * 0.3
                        force_y = perp[1] * force_magnitude * 0.7 + avoid_vector[1] * force_magnitude * 0.3
                    else:
                        # 对于人类保持原有比例
                        force_x = perp[0] * force_magnitude
                        force_y = perp[1] * force_magnitude
                    
                    # 添加额外的目标导向力，当人类静止且正在避让其他机器人软管时
                    if humans_stable_at_goal and is_robot_hose:
                        # 计算朝向目标的向量
                        goal_dir = np.array([self.gx - self.px, self.gy - self.py])
                        goal_dist = np.linalg.norm(goal_dir)
                        
                        if goal_dist > 0.001:
                            goal_dir = goal_dir / goal_dist
                            
                            # 计算避让方向与目标方向的点积，判断是否冲突
                            direction_conflict = -np.dot(avoid_vector, goal_dir)
                            
                            # 如果避让方向与目标方向高度冲突，适当降低避让力度
                            if direction_conflict > 0.7:  # 角度接近
                                # 减弱避让力以允许更好地朝向目标
                                force_x *= 0.7
                                force_y *= 0.7
                    
                    logging.debug(f"Hose safety force: ({force_x:.2f}, {force_y:.2f}), risk: {collision_risk:.2f}, is_robot_hose: {is_robot_hose}")
        except Exception as e:
            logging.debug(f"Error calculating hose safety force: {e}")
            
        return force_x, force_y, hose_danger
