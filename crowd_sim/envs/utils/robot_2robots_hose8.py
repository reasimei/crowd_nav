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
        5. 对软管凸起区域的特殊避让
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
        
        # 检查是否人类已经全部达到目标
        humans_stable_at_goal = False
        if hasattr(self.env, 'all_humans_at_goal') and self.env.all_humans_at_goal:
            if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                if (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold:
                    humans_stable_at_goal = True
        
        # 1. 设置安全距离 - 根据人类状态调整
        robot_safe_dist = 1.5 if humans_stable_at_goal else 2.0
        human_safe_dist = 0.8 if humans_stable_at_goal else 1.0
        hose_safe_dist = 1.2 if humans_stable_at_goal else 1.5   # 与软管安全距离
        
        # 2. 检查与其他机器人的碰撞
        for other_robot in ob:
            if isinstance(other_robot, Robot) and other_robot != self:
                other_pos = np.array([other_robot.px, other_robot.py])
                robot_dist = np.linalg.norm(other_pos - cur_pos)
                
                # 如果靠近另一个机器人，增加避让力
                if robot_dist < robot_safe_dist:
                    # 计算避障方向
                    avoid_direction = (cur_pos - other_pos) / max(0.1, robot_dist)
                    
                    # 避让力度随距离减小而增加
                    force_mag = (robot_safe_dist - robot_dist) / robot_safe_dist * 1.5
                    
                    # 应用避让力
                    modified_direction += avoid_direction * force_mag
                    need_to_avoid = True
                    
                    # 距离越近，减速越多
                    if robot_dist < robot_safe_dist * 0.5:
                        speed *= 0.5  # 严重降低速度
                    else:
                        speed *= 0.8  # 轻微降低速度
        
        # 3. 检查与软管的碰撞 - 增强对凸起的避让
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
                # 获取软管碰撞风险
                collision_risk, closest_entity, min_distance = self.predict_hose_collision(
                    [h for h in ob if not isinstance(h, Robot)]
                )
                
                # 如果有软管碰撞风险
                if collision_risk > 0.3:  # 设置一个有意义的阈值
                    need_to_avoid = True
                    
                    # 根据碰撞风险调整速度
                    speed *= max(0.3, 1.0 - collision_risk)
                    
                    if closest_entity is not None:
                        # 计算避开方向
                        entity_pos = np.array([closest_entity.px, closest_entity.py])
                        avoid_vec = cur_pos - entity_pos
                        avoid_dist = np.linalg.norm(avoid_vec)
                        
                        if avoid_dist > 0.001:
                            avoid_direction = avoid_vec / avoid_dist
                            avoid_force = avoid_direction * collision_risk * 2.0  # 增强避让力度
                            modified_direction += avoid_force
                
                # 额外检查每对软管组合 (更精确的避让，特别关注凸起区域)
                for i in range(0, len(self.env.robots), 2):
                    if i != self.robot_index and i+1 < len(self.env.robots):
                        # 获取其他软管对的两个机器人
                        other_robot1 = self.env.robots[i]
                        other_robot2 = self.env.robots[i+1]
                        
                        # 估计其他软管对的凸起程度
                        other_bulge_center, other_bulge_radius, other_bulge_factor = other_robot1.estimate_hose_bulge(other_robot2)
                        
                        # 计算软管线段
                        other_hose_seg_start = np.array([other_robot1.px, other_robot1.py])
                        other_hose_seg_end = np.array([other_robot2.px, other_robot2.py])
                        
                        # 基础软管安全距离增加对凸起的考虑
                        adjusted_safe_dist = hose_safe_dist
                        if other_bulge_factor > 0.3:  # 明显凸起时增加安全距离
                            adjusted_safe_dist += other_bulge_radius * 0.5  # 根据凸起程度增加安全距离
                        
                        # 计算当前机器人到其他软管段的距离
                        try:
                            from crowd_sim.envs.utils.utils import point_to_segment
                            distance = point_to_segment(cur_pos, other_hose_seg_start, other_hose_seg_end) - self.radius - self.env.hose_thickness
                        except:
                            # 如果专用函数不可用，使用简单计算
                            from crowd_sim.envs.utils.utils import point_to_hose_curve
                            distance = point_to_hose_curve(
                                (self.px, self.py),
                                (other_robot1.px, other_robot1.py),
                                (other_robot2.px, other_robot2.py),
                                self.env.hose_length
                            ) - self.radius - self.env.hose_thickness
                        
                        # 计算到凸起区域的距离（如果凸起明显）
                        if other_bulge_factor > 0.2:
                            # 计算到凸起中心的距离
                            dist_to_bulge = np.linalg.norm(cur_pos - other_bulge_center) - self.radius - self.env.hose_thickness
                            
                            # 如果在凸起范围内，调整距离估计
                            if dist_to_bulge < other_bulge_radius:
                                # 越靠近凸起中心，距离越小
                                bulge_distance = dist_to_bulge * (1.0 - other_bulge_factor * 0.8)
                                distance = min(distance, bulge_distance)
                        
                        # 如果距离太近，避让
                        if distance < adjusted_safe_dist:
                            need_to_avoid = True
                            
                            # 对凸起区域的特殊避让策略
                            if other_bulge_factor > 0.2 and np.linalg.norm(cur_pos - other_bulge_center) < other_bulge_radius * 1.5:
                                # 如果接近凸起区域，直接远离凸起中心
                                avoid_vec = cur_pos - other_bulge_center
                                avoid_dist = np.linalg.norm(avoid_vec)
                                
                                if avoid_dist > 0.001:
                                    avoid_direction = avoid_vec / avoid_dist
                                    # 凸起程度越高，避让力度越大
                                    avoid_magnitude = 2.5 * other_bulge_factor
                                    modified_direction += avoid_direction * avoid_magnitude
                                
                                # 凸起区域减速更多
                                speed *= 0.4
                            else:
                                # 常规避让 - 找到软管上最近点，然后远离它
                                closest_point = point_to_segment(cur_pos, other_hose_seg_start, other_hose_seg_end, return_point=True)
                                
                                if closest_point is not None:
                                    avoid_vec = cur_pos - closest_point
                                    avoid_dist = np.linalg.norm(avoid_vec)
                                    
                                    if avoid_dist > 0.001:
                                        avoid_direction = avoid_vec / avoid_dist
                                        # 避让力随距离减小而增加
                                        avoid_magnitude = (adjusted_safe_dist - distance) / adjusted_safe_dist * 2.0
                                        modified_direction += avoid_direction * avoid_magnitude
                                
                                # 距离越近，减速越多
                                if distance < adjusted_safe_dist * 0.3:
                                    speed *= 0.4  # 严重降低速度
                                else:
                                    speed *= 0.7  # 中度降低速度
                
                # 检查软管交叉并避让
                if hasattr(self, 'hose_crossings') and self.hose_crossings:
                    for other_robot1, other_robot2 in self.hose_crossings:
                        # 如果检测到软管交叉，使用更激进的避让策略
                        need_to_avoid = True
                        
                        # 计算交叉点（近似）
                        partner = self.get_hose_partner()
                        if partner is not None:
                            # 估计对方软管的凸起程度
                            other_bulge_center, other_bulge_radius, other_bulge_factor = other_robot1.estimate_hose_bulge(other_robot2)
                            
                            # 我方软管的凸起程度
                            bulge_center, bulge_radius, bulge_factor = self.estimate_hose_bulge(partner)
                            
                            # 如果两方都有明显凸起，情况尤其危险
                            combined_bulge_factor = max(bulge_factor, other_bulge_factor)
                            
                            # 计算两条线段的"交叉点"
                            p1 = np.array([self.px, self.py])
                            p2 = np.array([partner.px, partner.py])
                            p3 = np.array([other_robot1.px, other_robot1.py])
                            p4 = np.array([other_robot2.px, other_robot2.py])
                            
                            # 计算交叉区域
                            # 如果有凸起，使用凸起中心；否则使用线段中点
                            if combined_bulge_factor > 0.3:
                                # 使用凸起中心作为避让点
                                if bulge_factor > other_bulge_factor:
                                    intersection_approx = bulge_center
                                else:
                                    intersection_approx = other_bulge_center
                            else:
                                # 找到交点附近
                                midpoint1 = (p1 + p2) / 2
                                midpoint2 = (p3 + p4) / 2
                                intersection_approx = (midpoint1 + midpoint2) / 2
                            
                            # 远离交点
                            avoid_vec = cur_pos - intersection_approx
                            avoid_dist = np.linalg.norm(avoid_vec)
                            
                            if avoid_dist > 0.001:
                                avoid_direction = avoid_vec / avoid_dist
                                # 使用更强的避让力，并考虑凸起因素
                                avoid_magnitude = 2.5 + combined_bulge_factor * 1.5  # 交叉点使用强力避让
                                modified_direction += avoid_direction * avoid_magnitude
                            
                            # 显著降低速度，交叉+凸起情况下更慢
                            speed *= max(0.2, 0.3 - combined_bulge_factor * 0.1)
            except Exception as e:
                logging.debug(f"Error in hose collision avoidance: {e}")
        
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
        
        # 5. 人类稳定时的目标导向增强
        if humans_stable_at_goal and dist_to_goal > 0.5:
            # 检查自己和伙伴之间的距离和方向
            partner = self.get_hose_partner()
            if partner is not None:
                # 估计软管凸起程度
                bulge_center, bulge_radius, bulge_factor = self.estimate_hose_bulge(partner)
                
                # 如果有明显凸起，需要先协调位置
                if bulge_factor > 0.3:
                    # 距离太近，尝试拉开一点距离以减少凸起
                    partner_vec = np.array([partner.px, partner.py]) - cur_pos
                    partner_dist = np.linalg.norm(partner_vec)
                    
                    # 计算理想距离（软管长度的一半）
                    hose_length = getattr(self.env, 'hose_length', 2.0)
                    ideal_dist = hose_length * 0.7  # 70%的软管长度，减少凸起但不完全拉直
                    
                    if partner_dist < ideal_dist * 0.7:  # 距离太近
                        # 计算远离伙伴的力
                        if partner_dist > 0.001:
                            away_dir = -partner_vec / partner_dist
                            # 力度随距离增加而减小
                            away_magnitude = 0.8 * (1.0 - partner_dist / ideal_dist)
                            modified_direction += away_dir * away_magnitude
                        
                        # 调整速度
                        speed *= 0.8
                    
            # 计算当前修改后方向与目标方向的点积，检查方向偏差程度
            direction_alignment = np.dot(modified_direction, direction)
            
            if direction_alignment < 0.5:  # 当方向偏离较大时
                # 增加目标导向力
                goal_force = direction * 0.8
                modified_direction += goal_force
            
            # 如果没有明显的避障需求，提高速度
            if not need_to_avoid:
                speed = min(preferred_speed * 1.2, preferred_speed)  # 最多提高20%速度
        
        # 6. 标准化方向向量
        if need_to_avoid or humans_stable_at_goal:
            norm = np.linalg.norm(modified_direction)
            if norm > 0:
                modified_direction = modified_direction / norm
        
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
        预测软管是否会与人类或其他机器人软管发生碰撞
        
        Args:
            human_states: 人类状态列表
            
        Returns:
            tuple: (collision_risk, closest_entity, min_distance)
                - collision_risk: 碰撞风险值 (0-1)
                - closest_entity: 最近的实体（人类或机器人）
                - min_distance: 最短距离
        """
        try:
            # 检查是否使用软管
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            if not use_hose:
                return 0, None, float('inf')
            
            # 获取软管伙伴
            partner = self.get_hose_partner()
            if partner is None:
                return 0, None, float('inf')
                
            # 获取软管长度
            hose_length = getattr(self.env, 'hose_length', 0)
            if hose_length <= 0:
                return 0, None, float('inf')
                
            # 从环境中获取point_to_hose_curve函数
            from crowd_sim.envs.utils.utils import point_to_hose_curve
            
            # 估计当前软管的凸起程度
            bulge_center, bulge_radius, bulge_factor = self.estimate_hose_bulge(partner)
            
            # 检查每个人与软管的最短距离
            min_distance = float('inf')
            closest_entity = None
            
            # 1. 检查与人类的碰撞风险
            for human in human_states:
                # 计算人到软管的最短距离
                human_pos = (human.px, human.py)
                
                # 常规软管曲线距离
                distance = point_to_hose_curve(
                    human_pos, 
                    (self.px, self.py), 
                    (partner.px, partner.py), 
                    hose_length
                ) - human.radius - self.env.hose_thickness
                
                # 如果有明显凸起，考虑额外的凸起区域检测
                if bulge_factor > 0.2:  # 只有当凸起明显时才考虑
                    # 计算人到凸起中心的距离
                    bulge_dist = np.linalg.norm(np.array(human_pos) - bulge_center) - human.radius - self.env.hose_thickness
                    
                    # 如果人在凸起区域内，减少距离估计
                    if bulge_dist < bulge_radius:
                        # 距离修正：越接近凸起中心，距离越小
                        distance_adjust = bulge_dist * (1.0 - bulge_factor)
                        distance = min(distance, distance_adjust)
                
                # 更新距离跟踪
                self.last_human_hose_distances[human] = distance
                
                # 找到最小距离
                if distance < min_distance:
                    min_distance = distance
                    closest_entity = human
            
            # 2. 检查与其他机器人软管的碰撞风险 - 增强检测逻辑
            # 获取当前机器人的软管对索引
            pair_index = self.robot_index // 2
            
            # 检查所有其他软管对
            for i in range(0, len(self.env.robots), 2):
                other_pair_idx = i // 2
                if other_pair_idx != pair_index and i + 1 < len(self.env.robots):
                    other_robot1 = self.env.robots[i]
                    other_robot2 = self.env.robots[i+1]
                    
                    # 估计其他软管对的凸起情况
                    other_bulge_center, other_bulge_radius, other_bulge_factor = other_robot1.estimate_hose_bulge(other_robot2)
                    
                    # 方法1：使用现有的线段距离计算函数
                    points = [
                        np.array([self.px, self.py]),
                        np.array([partner.px, partner.py]),
                        np.array([other_robot1.px, other_robot1.py]),
                        np.array([other_robot2.px, other_robot2.py])
                    ]
                    
                    # 检查两软管端点之间的最短距离
                    hose_distance = self._min_distance_between_segments(
                        points[0], points[1], points[2], points[3]
                    ) - self.env.hose_thickness * 2  # 两条软管的厚度总和
                    
                    # 方法2：检查每个机器人到对方软管的距离（更保守的检测）
                    robot_to_hose_dist1 = point_to_hose_curve(
                        (self.px, self.py),
                        (other_robot1.px, other_robot1.py),
                        (other_robot2.px, other_robot2.py),
                        hose_length
                    ) - self.radius - self.env.hose_thickness
                    
                    robot_to_hose_dist2 = point_to_hose_curve(
                        (partner.px, partner.py),
                        (other_robot1.px, other_robot1.py),
                        (other_robot2.px, other_robot2.py),
                        hose_length
                    ) - partner.radius - self.env.hose_thickness
                    
                    other_robot_to_hose_dist1 = point_to_hose_curve(
                        (other_robot1.px, other_robot1.py),
                        (self.px, self.py),
                        (partner.px, partner.py),
                        hose_length
                    ) - other_robot1.radius - self.env.hose_thickness
                    
                    other_robot_to_hose_dist2 = point_to_hose_curve(
                        (other_robot2.px, other_robot2.py),
                        (self.px, self.py),
                        (partner.px, partner.py),
                        hose_length
                    ) - other_robot2.radius - self.env.hose_thickness
                    
                    # 考虑软管凸起因素 - 新增逻辑
                    # 如果我方软管有明显凸起
                    if bulge_factor > 0.2:
                        # 检查其他机器人与凸起区域的关系
                        dist_to_bulge1 = np.linalg.norm(np.array([other_robot1.px, other_robot1.py]) - bulge_center) - other_robot1.radius - self.env.hose_thickness
                        dist_to_bulge2 = np.linalg.norm(np.array([other_robot2.px, other_robot2.py]) - bulge_center) - other_robot2.radius - self.env.hose_thickness
                        
                        # 考虑凸起区域的影响
                        if dist_to_bulge1 < bulge_radius:
                            other_robot_to_hose_dist1 = min(other_robot_to_hose_dist1, dist_to_bulge1 * (1.0 - bulge_factor * 0.8))
                        
                        if dist_to_bulge2 < bulge_radius:
                            other_robot_to_hose_dist2 = min(other_robot_to_hose_dist2, dist_to_bulge2 * (1.0 - bulge_factor * 0.8))
                    
                    # 如果对方软管有明显凸起
                    if other_bulge_factor > 0.2:
                        # 检查我方机器人与对方凸起区域的关系
                        self_to_other_bulge = np.linalg.norm(np.array([self.px, self.py]) - other_bulge_center) - self.radius - self.env.hose_thickness
                        partner_to_other_bulge = np.linalg.norm(np.array([partner.px, partner.py]) - other_bulge_center) - partner.radius - self.env.hose_thickness
                        
                        # 考虑凸起区域的影响
                        if self_to_other_bulge < other_bulge_radius:
                            robot_to_hose_dist1 = min(robot_to_hose_dist1, self_to_other_bulge * (1.0 - other_bulge_factor * 0.8))
                        
                        if partner_to_other_bulge < other_bulge_radius:
                            robot_to_hose_dist2 = min(robot_to_hose_dist2, partner_to_other_bulge * (1.0 - other_bulge_factor * 0.8))
                    
                    # 取所有距离中的最小值作为软管间的距离
                    robot_hose_distance = min(robot_to_hose_dist1, robot_to_hose_dist2, 
                                             other_robot_to_hose_dist1, other_robot_to_hose_dist2)
                    
                    # 使用更保守的检测结果
                    hose_distance = min(hose_distance, robot_hose_distance)
                    
                    # 检查是否是新的最小距离
                    if hose_distance < min_distance:
                        min_distance = hose_distance
                        # 使用距离我们软管最近的机器人作为closest_entity
                        dist_to_robot1 = np.linalg.norm(np.array([self.px, self.py]) - np.array([other_robot1.px, other_robot1.py]))
                        dist_to_robot2 = np.linalg.norm(np.array([self.px, self.py]) - np.array([other_robot2.px, other_robot2.py]))
                        closest_entity = other_robot1 if dist_to_robot1 < dist_to_robot2 else other_robot2
            
            # 3. 计算软管交叉信息
            self.hose_crossings = []  # 重置交叉信息
            
            # 检测软管之间是否相交（判断线段是否相交）
            for i in range(0, len(self.env.robots), 2):
                other_pair_idx = i // 2
                if other_pair_idx != pair_index and i + 1 < len(self.env.robots):
                    other_robot1 = self.env.robots[i]
                    other_robot2 = self.env.robots[i+1]
                    
                    # 判断两线段是否相交
                    p1 = np.array([self.px, self.py])
                    p2 = np.array([partner.px, partner.py])
                    p3 = np.array([other_robot1.px, other_robot1.py])
                    p4 = np.array([other_robot2.px, other_robot2.py])
                    
                    # 检查线段相交
                    if self._segments_intersect(p1, p2, p3, p4):
                        self.hose_crossings.append((other_robot1, other_robot2))
                        # 如果发现相交，增加碰撞风险
                        if min_distance > 0.1:  # 确保不会将min_distance设置得太小
                            min_distance = 0.1
            
            # 4. 计算最终碰撞风险
            collision_risk = 0
            if min_distance < 2.0:
                collision_risk = max(0, 1.0 - min_distance / 2.0)
                
            # 人类稳定在目标后，仍保持对其他软管的高警觉
            humans_stable_at_goal = False
            if hasattr(self.env, 'all_humans_at_goal') and self.env.all_humans_at_goal:
                if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                    if (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold:
                        humans_stable_at_goal = True
            
            # 5. 凸起因子影响风险评估 - 新增逻辑
            # 如果当前软管有明显凸起，增加碰撞风险
            if bulge_factor > 0.3:  # 只考虑明显的凸起
                # 凸起因子越大，风险越高
                bulge_risk = bulge_factor * 0.3  # 最多增加0.3的风险
                collision_risk = min(1.0, collision_risk + bulge_risk)
            
            # 如果人类已经稳定，且交叉的软管数量大于0，增加碰撞风险
            if humans_stable_at_goal and len(self.hose_crossings) > 0:
                # 当人类稳定后，让机器人更重视软管交叉的风险
                extra_risk = 0.3 * len(self.hose_crossings)  # 每个交叉点增加0.3的风险
                collision_risk = min(1.0, collision_risk + extra_risk)
            
            return collision_risk, closest_entity, min_distance
        except Exception as e:
            logging.debug(f"Error predicting hose collision: {e}")
            return 0, None, float('inf')
    
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

    def _segments_intersect(self, p1, p2, p3, p4):
        """
        检查两条线段是否相交
        
        Args:
            p1, p2: 第一条线段的端点
            p3, p4: 第二条线段的端点
            
        Returns:
            bool: 线段是否相交
        """
        def ccw(a, b, c):
            # 判断三点是否逆时针排列
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        # 两线段相交当且仅当：
        # 1. p1,p2,p3 的方向与 p1,p2,p4 的方向相反
        # 2. p3,p4,p1 的方向与 p3,p4,p2 的方向相反
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _calculate_hose_safety_force(self, human_states):
        """
        计算保持软管安全的力向量，特别考虑软管凸起区域
        
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
            
            # 获取伙伴和软管凸起信息
            partner = self.get_hose_partner()
            if partner is None:
                return force_x, force_y, hose_danger
                
            # 估计软管凸起程度
            bulge_center, bulge_radius, bulge_factor = self.estimate_hose_bulge(partner)
            
            # 获取碰撞风险信息
            collision_risk, closest_entity, min_distance = self.predict_hose_collision(human_states)
            
            # 如果有碰撞风险
            if collision_risk > 0:
                if closest_entity is None:
                    return force_x, force_y, hose_danger
                
                # 计算软管中点
                hose_midpoint = np.array([
                    (self.px + partner.px) / 2, 
                    (self.py + partner.py) / 2
                ])
                
                # 判断最近实体是人还是机器人
                is_human = not isinstance(closest_entity, Robot)
                
                # 计算实体到软管中点的向量
                entity_to_midpoint = hose_midpoint - np.array([closest_entity.px, closest_entity.py])
                
                # 归一化
                dist = np.linalg.norm(entity_to_midpoint)
                if dist > 0.001:
                    entity_to_midpoint = entity_to_midpoint / dist
                
                # 计算软管方向
                hose_direction = np.array([partner.px - self.px, partner.py - self.py])
                hose_length = np.linalg.norm(hose_direction)
                
                if hose_length > 0.001:
                    hose_direction = hose_direction / hose_length
                    
                    # 计算垂直于软管的方向 (两个方向，取与实体方向相反的一个)
                    perp1 = np.array([-hose_direction[1], hose_direction[0]])
                    perp2 = -perp1
                    
                    # 选择与实体方向相反的垂直方向
                    if np.dot(perp1, entity_to_midpoint) > np.dot(perp2, entity_to_midpoint):
                        perp = perp1
                    else:
                        perp = perp2
                    
                    # 如果有凸起，考虑实体到凸起中心的方向
                    if bulge_factor > 0.3 and not is_human:
                        # 计算实体到凸起中心的向量
                        entity_to_bulge = bulge_center - np.array([closest_entity.px, closest_entity.py])
                        bulge_dist = np.linalg.norm(entity_to_bulge)
                        
                        if bulge_dist > 0.001:
                            entity_to_bulge = entity_to_bulge / bulge_dist
                            
                            # 混合垂直方向和凸起方向
                            # 凸起程度越高，越倾向于使用凸起方向
                            perp = perp * (1.0 - bulge_factor * 0.7) + entity_to_bulge * (bulge_factor * 0.7)
                            # 归一化
                            perp_norm = np.linalg.norm(perp)
                            if perp_norm > 0.001:
                                perp = perp / perp_norm
                    
                    # 根据风险程度计算力度
                    # 当人类静止时降低安全力度 - 但根据碰撞对象区分对待
                    force_magnitude = collision_risk * (1.0 if is_human else 2.0)
                    
                    if humans_stable_at_goal:
                        if is_human:
                            # 人类静止时，大幅减少对人类的避让力度（因为他们不动）
                            force_magnitude *= 0.3
                        else:
                            # 对机器人软管的避让力度仅略微减少（因为它们仍在移动）
                            force_magnitude *= 0.8
                            
                            # 如果检测到软管交叉，额外增加力度
                            if hasattr(self, 'hose_crossings') and self.hose_crossings:
                                # 增加额外的避让力度
                                force_magnitude *= 1.5
                    
                    # 考虑凸起因素 - 凸起程度越高，力度越大
                    if not is_human and bulge_factor > 0.2:
                        force_magnitude *= (1.0 + bulge_factor * 0.5)
                    
                    # 如果距离非常近，标记为危险
                    if min_distance < 0.3:
                        hose_danger = True
                        # 即使人类静止，非常接近时仍保持较高的力度（但人类与机器人区分对待）
                        if is_human:
                            force_magnitude = max(force_magnitude, 0.8 if humans_stable_at_goal else 1.2)
                        else:
                            # 对机器人软管保持较高避让力度，凸起时更高
                            base_force = 1.5
                            if bulge_factor > 0.3:
                                base_force = 1.8
                            force_magnitude = max(force_magnitude, base_force)
                    
                    # 计算最终力向量
                    # 基础方向是垂直于软管方向
                    force_vec = perp * force_magnitude
                    
                    # 添加一些朝向目标的倾向（尤其是人类已经静止的情况）
                    if humans_stable_at_goal:
                        # 计算朝向目标的单位向量
                        goal_dir = np.array([self.gx - self.px, self.gy - self.py])
                        goal_dist = np.linalg.norm(goal_dir)
                        
                        if goal_dist > 0.001:
                            goal_dir = goal_dir / goal_dist
                            
                            # 计算软管力与目标方向的点积
                            alignment = np.dot(force_vec, goal_dir)
                            
                            # 如果力方向与目标方向冲突较大，适当减弱
                            if alignment < -0.5:  # 力方向与目标方向有较大冲突
                                # 减弱避让力度，给目标方向更多权重
                                force_vec *= 0.6
                                
                                # 如果是对其他软管的避让，添加一个小的目标方向成分
                                if not is_human:
                                    force_vec += goal_dir * 0.3
                    
                    force_x = force_vec[0]
                    force_y = force_vec[1]
                    
                    logging.debug(f"Hose safety force: ({force_x:.2f}, {force_y:.2f}), risk: {collision_risk:.2f}")
            
            # 附加：处理自身软管凸起的协调力 - 新增逻辑
            if bulge_factor > 0.4 and humans_stable_at_goal:
                # 当有明显凸起时，对应坐标需要保持一定距离
                # 计算与伙伴之间的向量
                partner_vec = np.array([partner.px, partner.py]) - np.array([self.px, self.py])
                partner_dist = np.linalg.norm(partner_vec)
                
                # 获取软管长度
                hose_length = getattr(self.env, 'hose_length', 2.0)
                
                # 计算理想距离（软管长度的60-80%，减少凸起）
                ideal_dist = hose_length * 0.7
                
                # 如果距离太近，添加一个拉开的力
                if partner_dist < ideal_dist * 0.6 and partner_dist > 0.001:
                    # 力方向：远离伙伴
                    away_dir = -partner_vec / partner_dist
                    
                    # 力度：随距离增大而减小
                    away_magnitude = 0.6 * (1.0 - partner_dist / ideal_dist)
                    
                    # 添加到总力中
                    coordination_force = away_dir * away_magnitude
                    force_x += coordination_force[0]
                    force_y += coordination_force[1]
                    
                    logging.debug(f"Added coordination force to reduce bulge: ({coordination_force[0]:.2f}, {coordination_force[1]:.2f})")
                
        except Exception as e:
            logging.debug(f"Error calculating hose safety force: {e}")
            
        return force_x, force_y, hose_danger

    def estimate_hose_bulge(self, partner):
        """
        估计当前软管的凸起程度，根据两个机器人之间的距离与软管长度的比例计算
        
        Args:
            partner: 软管伙伴机器人
            
        Returns:
            tuple: (bulge_center, bulge_radius, bulge_factor)
                - bulge_center: 凸起中心位置
                - bulge_radius: 凸起半径
                - bulge_factor: 凸起程度系数 (0-1，0表示无凸起，1表示最大凸起)
        """
        if partner is None:
            return None, 0, 0
        
        # 获取机器人位置
        p1 = np.array([self.px, self.py])
        p2 = np.array([partner.px, partner.py])
        
        # 计算两机器人间的距离
        dist = np.linalg.norm(p2 - p1)
        
        # 获取软管长度
        hose_length = getattr(self.env, 'hose_length', 2.0)
        
        # 计算凸起因子：距离越近，凸起越明显
        # 当距离接近软管长度时，凸起很小；当距离很小时，凸起很大
        if dist >= hose_length:
            # 软管拉直，无凸起
            bulge_factor = 0.0
        else:
            # 使用非线性公式计算凸起程度
            # 当距离为0时，bulge_factor为1
            # 当距离为hose_length时，bulge_factor为0
            bulge_factor = 1.0 - (dist / hose_length)**0.5
        
        # 计算凸起中心位置（两机器人的中点）
        bulge_center = (p1 + p2) / 2.0
        
        # 计算凸起半径：根据软管长度和两机器人间距离计算
        # 使用弦高公式：h = r - (d/2)²/r，其中h是弦高，r是圆半径，d是弦长
        # 简化计算：当两机器人距离为0时，凸起半径为软管长度/2
        # 当两机器人距离接近软管长度时，凸起半径接近无穷大（趋于直线）
        if dist < hose_length * 0.99:  # 避免精度问题
            # 近似计算：使用软管长度一半作为基准，按照凸起因子调整
            bulge_radius = (hose_length / 2.0) * bulge_factor
            
            # 非常近时限制最大半径
            if bulge_radius > hose_length:
                bulge_radius = hose_length
        else:
            # 几乎拉直的情况
            bulge_radius = 0.0
        
        return bulge_center, bulge_radius, bulge_factor
