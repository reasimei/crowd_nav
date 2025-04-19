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
            action = self.policy.predict(state, self.robot_index)
            
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
            from crowd_nav.parser import args
            if args.policy == 'sarl' or args.policy=='h_sarl':
                base_action = self.policy.predict(state)
            if args.policy == 'llm_sarl'or args.policy=='h_llm_sarl':
                base_action = self.policy.predict(state, self.robot_index)
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

    def act_avoid_robots(self, ob, flag):
        """
        增强的机器人避让策略，添加先导-跟随队形和交叉区域协调通过机制
        """
        # 获取当前位置和目标位置
        cur_pos = np.array([self.px, self.py])
        goal_pos = np.array([self.gx, self.gy])
        
        # 初始化速度和方向
        preferred_speed = self.v_pref
        speed = preferred_speed
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
        
        # 如果人类没有稳定在目标点，使用原来的避让策略
        if not humans_stable_at_goal:
            # 设置安全距离
            robot_safe_dist = 1.5
            human_safe_dist = 1.0
            hose_safe_dist = 1.5
            
            # 检查与其他机器人的碰撞
            for other_robot in ob:
                if isinstance(other_robot, Robot) and other_robot != self:
                    other_pos = np.array([other_robot.px, other_robot.py])
                    robot_dist = np.linalg.norm(other_pos - cur_pos)
                    
                    if robot_dist < robot_safe_dist:
                        avoid_direction = (cur_pos - other_pos) / max(0.1, robot_dist)
                        force_mag = (robot_safe_dist - robot_dist) / robot_safe_dist * 1.5
                        modified_direction += avoid_direction * force_mag
                        need_to_avoid = True
                        speed *= 0.7
            
            # 检查与人类的碰撞
            for human in ob:
                if not isinstance(human, Robot):
                    human_pos = np.array([human.px, human.py])
                    human_dist = np.linalg.norm(human_pos - cur_pos)
                    
                    if human_dist < human_safe_dist:
                        avoid_direction = (cur_pos - human_pos) / max(0.1, human_dist)
                        force_mag = (human_safe_dist - human_dist) / human_safe_dist * 1.5
                        modified_direction += avoid_direction * force_mag
                        need_to_avoid = True
                        speed *= 0.5
            
            # 软管避让逻辑
            if hasattr(self, 'predict_hose_collision'):
                collision_risk, closest_entity, min_distance = self.predict_hose_collision(
                    [h for h in ob if not isinstance(h, Robot)]
                )
                
                if collision_risk > 0.3:
                    need_to_avoid = True
                    speed *= max(0.4, 1.0 - collision_risk)
            
            # 标准化方向
            if need_to_avoid:
                norm = np.linalg.norm(modified_direction)
                if norm > 0:
                    modified_direction = modified_direction / norm
        
        # 如果人类稳定在目标点，启用先导-跟随协调机制
        else:
            # 获取伙伴和角色
            partner = self.get_hose_partner()
            if partner is not None:
                lead, follow, is_lead = self.identify_lead_follow_roles()
                
                # 检测交叉区域
                in_crossing_zone, center_distance, others_in_zone = self.detect_crossing_zone()
                priority, should_wait = self.calculate_crossing_priority()
                
                # 确定移动方向和对齐方式
                movement_dir = self.determine_movement_direction()
                alignment_type = self.determine_alignment_type()
                
                # 记录状态用于调试
                self.is_lead = is_lead
                self.movement_direction = movement_dir
                self.alignment_type = alignment_type
                self.in_crossing_zone = in_crossing_zone
                self.crossing_priority = priority
                
                # 先导机器人逻辑
                if is_lead:
                    # 如果在交叉区域且应该等待，减速或停止
                    if in_crossing_zone and should_wait:
                        speed *= 0.3  # 大幅减速
                        
                        # 如果已经很近中心，几乎停止
                        if center_distance < 2.5:
                            speed *= 0.1
                            
                            # 记录等待状态
                            self.is_waiting = True
                            logging.debug(f"Robot {self.robot_index} (lead) waiting in crossing zone")
                        else:
                            # 小心接近
                            self.is_waiting = False
                        
                    # 如果是我们优先通过，保持正常速度
                    elif in_crossing_zone and priority > 5:
                        # 小幅减速，保持警觉
                        speed *= 0.8
                        self.is_waiting = False
                        logging.debug(f"Robot {self.robot_index} (lead) passing through with priority {priority}")
                    else:
                        # 正常移动
                        self.is_waiting = False
                    
                    # 检查软管碰撞风险
                    if hasattr(self, 'predict_hose_collision'):
                        collision_risk, closest_entity, min_distance = self.predict_hose_collision(
                            [h for h in ob if not isinstance(h, Robot)]
                        )
                        
                        if collision_risk > 0.3:
                            need_to_avoid = True
                            # 调整避让力度
                            avoid_strength = collision_risk * 1.5
                            
                            if closest_entity is not None:
                                # 计算避障方向
                                entity_pos = np.array([closest_entity.px, closest_entity.py])
                                avoid_vec = cur_pos - entity_pos
                                
                                if np.linalg.norm(avoid_vec) > 0.001:
                                    avoid_vec = avoid_vec / np.linalg.norm(avoid_vec)
                                    modified_direction += avoid_vec * avoid_strength
                            
                            # 碰撞风险高时减速
                            speed *= max(0.4, 1.0 - collision_risk)
                    
                    # 考虑其他机器人避让
                    for other_robot in ob:
                        if isinstance(other_robot, Robot) and other_robot != self and other_robot != partner:
                            other_pos = np.array([other_robot.px, other_robot.py])
                            robot_dist = np.linalg.norm(other_pos - cur_pos)
                            
                            # 设定机器人安全距离
                            robot_safe_dist = 1.2
                            
                            if robot_dist < robot_safe_dist:
                                # 计算避障方向
                                avoid_direction = (cur_pos - other_pos) / max(0.1, robot_dist)
                                
                                # 力度随距离减小而增加
                                force_mag = (robot_safe_dist - robot_dist) / robot_safe_dist * 1.2
                                
                                # 应用避让力
                                modified_direction += avoid_direction * force_mag
                                need_to_avoid = True
                                
                                # 减速
                                speed *= 0.8
                
                # 跟随者逻辑
                else:
                    # 跟随者完全听从先导，保持队形
                    form_force_x, form_force_y, formation_distance = self.maintain_formation(lead)
                    
                    # 使用队形力替代目标方向
                    if formation_distance > 0.3:  # 当偏离队形较大时
                        # 队形力对方向的影响随偏离程度增加
                        formation_weight = min(0.8, formation_distance * 0.2)
                        
                        # 计算加权方向
                        modified_direction = (1.0 - formation_weight) * direction
                        modified_direction += formation_weight * np.array([form_force_x, form_force_y])
                        
                        # 标准化
                        norm = np.linalg.norm(modified_direction)
                        if norm > 0:
                            modified_direction = modified_direction / norm
                        
                        # 根据偏离程度调整速度
                        deviation_factor = min(1.0, formation_distance / 3.0)
                        speed_scale = 0.7 + 0.3 * deviation_factor  # 偏离越大速度越快(0.7-1.0)
                        speed *= speed_scale
                        
                        logging.debug(f"Robot {self.robot_index} (follower) maintaining formation, distance: {formation_distance:.2f}")
                    else:
                        # 队形良好，与先导机器人保持一致速度
                        if hasattr(lead, 'policy') and hasattr(lead.policy, 'last_velocity'):
                            lead_speed = np.linalg.norm(lead.policy.last_velocity)
                            speed = min(speed, lead_speed * 1.1)  # 略快于先导以保持队形
                        
                        # 如果先导在等待，跟随者也等待
                        if hasattr(lead, 'is_waiting') and lead.is_waiting:
                            speed *= 0.2  # 几乎停止
                            self.is_waiting = True
                        else:
                            self.is_waiting = False
                    
                    # 跟随者也需要避开其他非组队机器人
                    for other_robot in ob:
                        if isinstance(other_robot, Robot) and other_robot != self and other_robot != lead:
                            other_pos = np.array([other_robot.px, other_robot.py])
                            robot_dist = np.linalg.norm(other_pos - cur_pos)
                            
                            # 减小安全距离，以免过度避让
                            robot_safe_dist = 0.8
                            
                            if robot_dist < robot_safe_dist:
                                # 较轻微的避让
                                avoid_direction = (cur_pos - other_pos) / max(0.1, robot_dist)
                                force_mag = (robot_safe_dist - robot_dist) / robot_safe_dist
                                
                                # 减小避让力对方向的影响
                                modified_direction = 0.7 * modified_direction + 0.3 * avoid_direction * force_mag
                                need_to_avoid = True
                                
                                # 标准化
                                norm = np.linalg.norm(modified_direction)
                                if norm > 0:
                                    modified_direction = modified_direction / norm
                                
                                # 轻微减速
                                speed *= 0.9
        
        # 标准化方向
        if need_to_avoid:
            norm = np.linalg.norm(modified_direction)
            if norm > 0:
                modified_direction = modified_direction / norm
        
        # 计算最终速度分量
        vx = speed * modified_direction[0]
        vy = speed * modified_direction[1]
        
        # 保存最后速度用于协调
        if not hasattr(self.policy, 'last_velocity'):
            self.policy.last_velocity = np.array([vx, vy])
        else:
            self.policy.last_velocity = np.array([vx, vy])
        
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

    def determine_movement_direction(self):
        """
        根据起始点和目标点确定机器人对的主要移动方向
        
        Returns:
            str: 'horizontal' - 水平移动(左右)
                 'vertical' - 垂直移动(上下)
        """
        dx = abs(self.gx - self.px)
        dy = abs(self.gy - self.py)
        
        # 根据起点到目标点的主要方向确定
        if dx > dy:
            return 'horizontal'  # 左右移动
        else:
            return 'vertical'    # 上下移动

    def determine_alignment_type(self):
        """
        根据移动方向确定最佳对齐方式
        
        Returns:
            str: 'horizontal_alignment' - 横向对齐(左右排列)
                 'vertical_alignment' - 纵向对齐(上下排列)
        """
        movement_dir = self.determine_movement_direction()
        
        # 如果是水平方向移动，需要纵向对齐
        if movement_dir == 'horizontal':
            return 'vertical_alignment'  # 上下排列
        else:
            return 'horizontal_alignment'  # 左右排列

    def identify_lead_follow_roles(self):
        """
        确定当前机器人对中的先导和跟随角色
        
        Returns:
            tuple: (lead_robot, follow_robot, is_lead)
                - lead_robot: 先导机器人
                - follow_robot: 跟随机器人
                - is_lead: 当前机器人是否为先导角色
        """
        partner = self.get_hose_partner()
        if partner is None:
            return None, None, True  # 没有伙伴，默认自己是先导
        
        movement_dir = self.determine_movement_direction()
        
        # 水平移动情况 (左右移动)
        if movement_dir == 'horizontal':
            # 如果是从左到右移动
            if self.gx > self.px:
                # 比较y坐标，y较小的(上方的)先导
                if self.py < partner.py:
                    return self, partner, True
                else:
                    return partner, self, False
            # 如果是从右到左移动
            else:
                # 比较y坐标，y较小的(上方的)先导
                if self.py < partner.py:
                    return self, partner, True
                else:
                    return partner, self, False
        
        # 垂直移动情况 (上下移动)
        else:
            # 如果是从上到下移动
            if self.gy < self.py:
                # 比较x坐标，x较小的(左侧的)先导
                if self.px < partner.px:
                    return self, partner, True
                else:
                    return partner, self, False
            # 如果是从下到上移动
            else:
                # 比较x坐标，x较小的(左侧的)先导
                if self.px < partner.px:
                    return self, partner, True
                else:
                    return partner, self, False

    def detect_crossing_zone(self):
        """
        检测是否处于中心危险交叉区域
        
        Returns:
            tuple: (in_zone, center_distance, other_pairs_in_zone)
                - in_zone: 是否在交叉区域内
                - center_distance: 到中心的距离
                - other_pairs_in_zone: 其他机器人对在交叉区域内的数量
        """
        # 定义中心区域半径
        center_radius = 3.0
        
        # 计算到中心点的距离
        center_distance = np.linalg.norm(np.array([self.px, self.py]))
        in_zone = center_distance < center_radius
        
        # 检查其他机器人对是否在区域内
        other_pairs_in_zone = 0
        my_pair_index = self.robot_index // 2
        
        for i in range(0, len(self.env.robots), 2):
            other_pair_idx = i // 2
            if other_pair_idx != my_pair_index and i+1 < len(self.env.robots):
                robot1 = self.env.robots[i]
                robot2 = self.env.robots[i+1]
                
                r1_distance = np.linalg.norm(np.array([robot1.px, robot1.py]))
                r2_distance = np.linalg.norm(np.array([robot2.px, robot2.py]))
                
                if r1_distance < center_radius or r2_distance < center_radius:
                    other_pairs_in_zone += 1
        
        return in_zone, center_distance, other_pairs_in_zone

    def calculate_crossing_priority(self):
        """
        计算当前机器人对的通行优先级
        
        Returns:
            tuple: (priority, should_wait)
                - priority: 优先级值(0-10)，越高优先级越高
                - should_wait: 是否应该等待
        """
        in_zone, center_distance, other_pairs_in_zone = self.detect_crossing_zone()
        
        # 获取伙伴
        partner = self.get_hose_partner()
        if partner is None:
            return 5, False  # 没有伙伴，中等优先级，不等待
        
        # 如果不在交叉区域内且远离中心，高优先级
        if not in_zone and center_distance > 4.0:
            return 8, False
            
        # 获取移动方向和先导/跟随角色
        movement_dir = self.determine_movement_direction()
        lead, follow, is_lead = self.identify_lead_follow_roles()
        
        # 垂直移动的机器人对优先于水平移动的
        # 这是基于上传的截图场景设计的特定策略
        base_priority = 7 if movement_dir == 'vertical' else 4
        
        # 计算到目标的距离
        dist_to_goal = np.linalg.norm(np.array([self.gx - self.px, self.gy - self.py]))
        
        # 如果已经很接近目标，提高优先级
        if dist_to_goal < 2.0:
            base_priority += 2
        
        # 有其他机器人对在交叉区域时，根据移动方向判断是否等待
        if other_pairs_in_zone > 0:
            # 如果其他对是垂直移动(上下)且当前对是水平移动(左右)，应该等待
            other_vertical_moving = False
            
            # 检查其他机器人对的移动方向
            my_pair_index = self.robot_index // 2
            for i in range(0, len(self.env.robots), 2):
                other_pair_idx = i // 2
                if other_pair_idx != my_pair_index and i+1 < len(self.env.robots):
                    robot1 = self.env.robots[i]
                    if hasattr(robot1, 'determine_movement_direction'):
                        if robot1.determine_movement_direction() == 'vertical':
                            other_vertical_moving = True
                            break
            
            # 水平移动时，如果有垂直移动的对在交叉区域，等待
            if movement_dir == 'horizontal' and other_vertical_moving:
                return base_priority - 3, True
        
        # 返回最终优先级和等待标志
        return base_priority, False

    def calculate_formation_position(self, lead_robot):
        """
        计算跟随者应保持的位置，确保软管近似直线
        
        Args:
            lead_robot: 先导机器人
            
        Returns:
            tuple: (target_x, target_y) 目标位置
        """
        if lead_robot is None:
            return self.px, self.py
        
        # 获取对齐方式
        alignment = self.determine_alignment_type()
        
        # 理想距离(软管长度的大约80-90%)
        ideal_distance = getattr(self.env, 'hose_length', 2.0) * 0.85
        
        # 计算先导方向向量
        lead_dir = np.array([lead_robot.gx - lead_robot.px, lead_robot.gy - lead_robot.py])
        lead_dist = np.linalg.norm(lead_dir)
        
        if lead_dist > 0.001:
            lead_dir = lead_dir / lead_dist
        else:
            lead_dir = np.array([1.0, 0.0])  # 默认向右
        
        # 根据对齐方式计算偏移向量
        if alignment == 'horizontal_alignment':  # 左右对齐
            # 计算垂直于移动方向的向量(左右)
            offset_dir = np.array([-lead_dir[1], lead_dir[0]])
            # 根据当前位置确定偏移方向
            if np.dot(np.array([self.px - lead_robot.px, self.py - lead_robot.py]), offset_dir) < 0:
                offset_dir = -offset_dir
        else:  # 纵向对齐
            # 计算垂直于移动方向的向量(上下)
            offset_dir = np.array([lead_dir[1], -lead_dir[0]])
            # 根据当前位置确定偏移方向
            if np.dot(np.array([self.px - lead_robot.px, self.py - lead_robot.py]), offset_dir) < 0:
                offset_dir = -offset_dir
        
        # 计算目标位置: 先导位置 + 偏移距离
        target_x = lead_robot.px + offset_dir[0] * ideal_distance
        target_y = lead_robot.py + offset_dir[1] * ideal_distance
        
        return target_x, target_y

    def maintain_formation(self, lead_robot):
        """
        为跟随者计算保持队形的力，确保软管近似直线
        
        Args:
            lead_robot: 先导机器人
            
        Returns:
            tuple: (force_x, force_y, formation_distance)
                - force_x, force_y: 保持队形的力
                - formation_distance: 当前与理想队形的距离
        """
        if lead_robot is None:
            return 0, 0, 0
        
        # 计算理想位置
        target_x, target_y = self.calculate_formation_position(lead_robot)
        
        # 计算当前位置到理想位置的向量
        formation_vec = np.array([target_x - self.px, target_y - self.py])
        formation_distance = np.linalg.norm(formation_vec)
        
        # 近似软管长度
        hose_length = getattr(self.env, 'hose_length', 2.0)
        
        # 计算与先导机器人的当前距离
        current_dist = np.linalg.norm(np.array([
            lead_robot.px - self.px,
            lead_robot.py - self.py
        ]))
        
        # 计算队形保持力
        force_x, force_y = 0, 0
        
        if formation_distance > 0.1:  # 避免除以非常小的数
            # 力度随着偏离程度增加
            force_magnitude = min(2.0, formation_distance * 0.8)
            
            # 方向是朝向目标位置
            force_dir = formation_vec / formation_distance
            
            # 计算力的分量
            force_x = force_dir[0] * force_magnitude
            force_y = force_dir[1] * force_magnitude
            
            # 如果距离先导太远(软管绷紧)，增加力度
            if current_dist > hose_length * 0.95:
                tension_factor = min(3.0, (current_dist - hose_length * 0.95) * 4.0)
                
                # 朝向先导的方向
                to_lead_dir = np.array([
                    lead_robot.px - self.px,
                    lead_robot.py - self.py
                ])
                if np.linalg.norm(to_lead_dir) > 0.001:
                    to_lead_dir = to_lead_dir / np.linalg.norm(to_lead_dir)
                    
                    # 增加朝向先导的力
                    force_x += to_lead_dir[0] * tension_factor
                    force_y += to_lead_dir[1] * tension_factor
            
            # 如果距离先导太近，增加排斥力
            elif current_dist < hose_length * 0.4:
                repel_factor = min(2.0, (hose_length * 0.4 - current_dist) * 3.0)
                
                # 远离先导的方向
                from_lead_dir = np.array([
                    self.px - lead_robot.px,
                    self.py - lead_robot.py
                ])
                if np.linalg.norm(from_lead_dir) > 0.001:
                    from_lead_dir = from_lead_dir / np.linalg.norm(from_lead_dir)
                    
                    # 增加远离先导的力
                    force_x += from_lead_dir[0] * repel_factor
                    force_y += from_lead_dir[1] * repel_factor
        
        return force_x, force_y, formation_distance
