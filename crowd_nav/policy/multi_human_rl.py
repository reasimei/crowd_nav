import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL
from crowd_sim.envs.utils.state import ObservableState
import logging


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()
        self.warning_logged = False  # 添加警告标志


    def predict(self, state,robot_index):
        """
        根据policy-based RL算法选择操作
        """
        if self.phase is None or self.device is None:
            raise AttributeError('请先设置phase和device值!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('请在训练模式下设置epsilon值!')
        
        if self.reach_destination(state):
            return ActionXY(0, 0)  # 如果已到达目标点，停止移动
        
        # 检查人类是否全部到达目标
        all_humans_at_goal = False
        humans_stable_at_goal = False
        
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'all_humans_at_goal'):
                all_humans_at_goal = self.env.all_humans_at_goal
                # 检查人类是否稳定在目标位置
                if hasattr(self.env, 'humans_at_goal_time') and hasattr(self.env, 'goal_waiting_threshold'):
                    humans_stable_at_goal = all_humans_at_goal and (self.env.global_time - self.env.humans_at_goal_time) >= self.env.goal_waiting_threshold
            
        # 检查是否有软管伙伴
        use_hose = False
        has_hose_partner = False
        hose_partner = None
        
        try:
            from crowd_nav.parser import args
            use_hose = getattr(args, 'hose', False)
            
            # 检查是否有软管伙伴
            if use_hose and hasattr(self, 'env') and self.env is not None:
                if hasattr(state, 'robot_index'):
                    robot_index = state.robot_index
                    partner_idx = robot_index + 1 if robot_index % 2 == 0 else robot_index - 1
                    
                    if hasattr(self.env, 'robots') and 0 <= partner_idx < len(self.env.robots):
                        has_hose_partner = True
                        hose_partner = self.env.robots[partner_idx]
        except Exception as e:
            logging.debug(f"Error checking hose status: {e}")
        
        # 距离目标的距离
        dist_to_goal = np.linalg.norm([state.gx - state.px, state.gy - state.py])
        approaching_goal = dist_to_goal < 1.0

        state_tensor = self.transform(state)
        # 使用当前网络获取动作价值
        if self.phase == 'train' and np.random.random() < self.epsilon:
            # 训练时执行随机探索
            if approaching_goal and humans_stable_at_goal and np.random.random() < 0.7:
                # 如果接近目标且人类都已到达目标位置，有更高概率选择直接朝向目标的动作
                goal_dir = np.array([state.gx - state.px, state.gy - state.py])
                if np.linalg.norm(goal_dir) > 0:
                    goal_dir = goal_dir / np.linalg.norm(goal_dir)
                    # 找到最接近目标方向的低速动作
                    best_action = None
                    best_dir_match = -1
                    for action in self.action_space:
                        action_dir = np.array([action.vx, action.vy])
                        speed = np.linalg.norm(action_dir)
                        if speed < 0.01:  # 避免静止不动或速度过低的动作
                            continue
                        action_dir = action_dir / speed
                        dir_match = np.dot(goal_dir, action_dir)
                        # 更喜欢中速的动作
                        if dir_match > 0.8 and 0.2 < speed < 0.8:
                            if dir_match > best_dir_match:
                                best_dir_match = dir_match
                                best_action = action
                    
                    if best_action is not None:
                        max_action = best_action
                    else:
                        max_action = self.action_space[np.random.choice(len(self.action_space))]
                else:
                    max_action = self.action_space[np.random.choice(len(self.action_space))]
            else:
                max_action = self.action_space[np.random.choice(len(self.action_space))]
            max_action_value = None
        else:
            # 使用当前网络获取动作价值
            self.model.eval()
            with torch.no_grad():
                inputs = torch.Tensor([state_tensor]).to(self.device)
                outputs = self.model(inputs)
                # logging.debug(f'Action值: {outputs[0].cpu().numpy()}')
                
                # 如果接近目标点，筛选低速动作
                if approaching_goal and not humans_stable_at_goal:
                    outputs_np = outputs[0].cpu().numpy()
                    action_values = outputs_np.copy()
                    for i, action in enumerate(self.action_space):
                        if np.linalg.norm([action.vx, action.vy]) > 0.5:  # 如果速度太大
                            action_values[i] -= 1.0  # 降低其值
                    max_action_idx = np.argmax(action_values)
                else:
                    max_action_idx = outputs[0].argmax().item()
                max_action = self.action_space[max_action_idx]
                max_action_value = outputs[0][max_action_idx].item()
                
                # 调试信息
                if humans_stable_at_goal and approaching_goal:
                    logging.debug(f"接近目标 AND 人类已稳定到达目标. dist_to_goal={dist_to_goal:.2f}")
                    logging.debug(f"选择动作: vx={max_action.vx:.2f}, vy={max_action.vy:.2f}, action_value={max_action_value:.4f}")
        
        # 当人类稳定在目标位置且机器人距离目标较远时，增强朝向目标的动作倾向
        if humans_stable_at_goal and dist_to_goal > 0.5:
            goal_dir = np.array([state.gx - state.px, state.gy - state.py])
            if np.linalg.norm(goal_dir) > 0:
                goal_dir = goal_dir / np.linalg.norm(goal_dir)
                
                # 考虑替代动作
                for action in self.action_space:
                    action_dir = np.array([action.vx, action.vy])
                    speed = np.linalg.norm(action_dir)
                    
                    # 只考虑速度适中且朝向目标的动作
                    if speed > 0.3 and speed < 0.8:
                        action_dir = action_dir / speed
                        dir_match = np.dot(goal_dir, action_dir)
                        
                        if dir_match > 0.9:  # 方向高度匹配
                            # 计算到达目标所需时间估计
                            est_time_to_goal = dist_to_goal / speed
                            if est_time_to_goal < 10:  # 合理的到达时间
                                max_action = action
                                logging.debug(f"人类稳定，选择更直接的目标动作: vx={action.vx:.2f}, vy={action.vy:.2f}")
                                break
        
        # 当使用软管且人类稳定在目标时，协调与软管伙伴的行动
        if use_hose and has_hose_partner and humans_stable_at_goal and hose_partner is not None:
            try:
                # 计算伙伴距离其目标的距离
                partner_dist_to_goal = np.linalg.norm([
                    hose_partner.gx - hose_partner.px,
                    hose_partner.gy - hose_partner.py
                ])
                
                # 获取软管长度
                hose_length = getattr(self.env, 'hose_length', 0)
                
                # 计算当前软管长度
                current_hose_length = np.linalg.norm([
                    hose_partner.px - state.px,
                    hose_partner.py - state.py
                ])
                
                # 计算预期的方向向量
                robot_to_goal = np.array([state.gx - state.px, state.gy - state.py])
                if np.linalg.norm(robot_to_goal) > 0:
                    robot_to_goal = robot_to_goal / np.linalg.norm(robot_to_goal)
                
                partner_to_robot = np.array([state.px - hose_partner.px, state.py - hose_partner.py])
                if np.linalg.norm(partner_to_robot) > 0:
                    partner_to_robot = partner_to_robot / np.linalg.norm(partner_to_robot)
                
                # 检查是否会过度拉伸软管
                action_speed = np.linalg.norm([max_action.vx, max_action.vy])
                new_pos_x = state.px + max_action.vx
                new_pos_y = state.py + max_action.vy
                new_hose_length = np.linalg.norm([
                    hose_partner.px - new_pos_x,
                    hose_partner.py - new_pos_y
                ])
                
                # 调整动作避免过度拉伸
                if new_hose_length > hose_length * 1.1:
                    # 寻找不会拉伸的替代动作
                    for action in self.action_space:
                        test_new_x = state.px + action.vx
                        test_new_y = state.py + action.vy
                        test_hose_length = np.linalg.norm([
                            hose_partner.px - test_new_x,
                            hose_partner.py - test_new_y
                        ])
                        
                        if test_hose_length <= hose_length * 1.05:
                            # 检查是否朝向目标
                            action_dir = np.array([action.vx, action.vy])
                            if np.linalg.norm(action_dir) > 0:
                                action_dir = action_dir / np.linalg.norm(action_dir)
                                if np.dot(action_dir, robot_to_goal) > 0.5:
                                    max_action = action
                                    logging.debug(f"调整动作避免软管拉伸: vx={action.vx:.2f}, vy={action.vy:.2f}")
                                    break
            except Exception as e:
                logging.debug(f"Error in hose partner coordination: {e}")
                
        # 平滑操作，避免抖动
        if hasattr(self, 'last_action') and self.last_action is not None:
            # 如果人类已经稳定达到目标，减少平滑权重，允许更加灵活的动作
            smoothing_factor = 0.3 if not humans_stable_at_goal else 0.15
            action_change = np.linalg.norm([
                max_action.vx - self.last_action.vx,
                max_action.vy - self.last_action.vy
            ])
            
            # 如果动作变化过大且不处于接近目标状态
            if action_change > 0.3 and not (approaching_goal and humans_stable_at_goal):
                # 平滑动作
                vx = (1 - smoothing_factor) * max_action.vx + smoothing_factor * self.last_action.vx
                vy = (1 - smoothing_factor) * max_action.vy + smoothing_factor * self.last_action.vy
                smoothed_action = ActionXY(vx, vy)
                logging.debug(f"平滑动作: 原始=[{max_action.vx:.2f}, {max_action.vy:.2f}], "
                             f"平滑后=[{smoothed_action.vx:.2f}, {smoothed_action.vy:.2f}]")
                max_action = smoothed_action
            
        # 根据距离目标的远近调整速度
        if approaching_goal:
            # 接近目标时减速
            speed = np.linalg.norm([max_action.vx, max_action.vy])
            if speed > 0:
                # 计算减速系数 - 越近速度越低
                speed_factor = max(0.3, min(1.0, dist_to_goal / 1.0))
                
                # 如果人类已稳定，使用更高的速度系数
                if humans_stable_at_goal:
                    speed_factor = max(0.5, min(1.0, dist_to_goal / 0.8))
                
                direction = np.array([max_action.vx, max_action.vy]) / speed
                adjusted_speed = speed * speed_factor
                max_action = ActionXY(direction[0] * adjusted_speed, direction[1] * adjusted_speed)
                logging.debug(f"接近目标，调整速度: 系数={speed_factor:.2f}, "
                             f"速度=[{max_action.vx:.2f}, {max_action.vy:.2f}]")
        
        # 保存当前动作用于下次平滑操作
        self.last_action = max_action
        
        return max_action

    def compute_reward(self, nav, humans, other_robots=None):
        # collision detection with humans
        human_reward = self.compute_human_avoidance_reward(nav, humans)
        
        # collision detection with other robots
        robot_reward = 0
        if other_robots:
            robot_reward = self.compute_robot_coordination_reward(nav, other_robots)
            
        return human_reward + robot_reward

    def compute_human_avoidance_reward(self, nav, humans):
        # 使用原有的奖励计算逻辑
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -2.5
        elif reaching_goal:
            reward = 10
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0
        return reward
    
    def compute_robot_coordination_reward(self, nav, other_robots):
        robot_reward = 0
        dmin = float('inf')
        collision = False
        
        for robot in other_robots:
            # distance from robot to other robot
            dx = nav.px - robot.px
            dy = nav.py - robot.py
            dist = (dx ** 2 + dy ** 2) ** 0.5
            
            if dist < nav.radius + robot.radius:
                collision = True
                break
            
            if dist < dmin:
                dmin = dist
                
        # collision penalty
        if collision:
            robot_reward = -0.25
        
        # reward for keeping a safe distance from other robots
        elif dmin < 1.5:  # use appropriate threshold
            # Penalty grows as robots get closer
            robot_reward = -0.1 * (1.5 - dmin)
            
        return robot_reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for value network
        :param state: a State instance or a list of State instances
        :return: tensor of shape (# humans, self.joint_state_dim)
        """
        if state is None:
            return torch.zeros(1, self.joint_state_dim, device=self.device)

        if isinstance(state, list):
            # For a list of states, transform each and concatenate
            return torch.cat([self.transform(s) for s in state])

        # Check if we need to standardize dimensions for input tensors
        def standardize_dim(tensor, target_dim=13):
            """Standardize tensor dimensions to the target dimension"""
            if tensor is None:
                return torch.zeros(1, target_dim, device=self.device)
                
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
                
            batch_size = tensor.shape[0]
            current_dim = tensor.shape[1] if len(tensor.shape) > 1 else 1
            
            if current_dim > target_dim:
                logging.debug(f"Truncating tensor from dim {current_dim} to {target_dim}")
                return tensor[:, :target_dim]
            elif current_dim < target_dim:
                logging.debug(f"Padding tensor from dim {current_dim} to {target_dim}")
                padding = torch.zeros(batch_size, target_dim - current_dim, device=self.device)
                return torch.cat([tensor, padding], dim=1)
            else:
                return tensor

        # Check if state is a Tensor already
        if isinstance(state, torch.Tensor):
            # Handle tensor input directly
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            state_tensor = standardize_dim(state)
            return state_tensor

        state_tensors = []
        
        # Handle human states
        if hasattr(state, 'human_states') and state.human_states:
            for human in state.human_states:
                if human is None:
                    continue
                human_tensor = torch.Tensor([human.px, human.py, human.vx, human.vy, human.radius]).to(self.device)
                state_tensors.append(standardize_dim(human_tensor))
        
        # Handle robot states            
        if hasattr(state, 'other_robot_states') and state.other_robot_states:
            for robot in state.other_robot_states:
                if robot is None:
                    continue
                robot_tensor = torch.Tensor([robot.px, robot.py, robot.vx, robot.vy, robot.radius,
                                            robot.gx, robot.gy, robot.v_pref, robot.theta]).to(self.device)
                state_tensors.append(standardize_dim(robot_tensor))
        
        # Handle self state
        if hasattr(state, 'self_state') and state.self_state:
            self_tensor = torch.Tensor([state.self_state.px, state.self_state.py, state.self_state.vx, state.self_state.vy,
                                        state.self_state.radius, state.self_state.gx, state.self_state.gy,
                                        state.self_state.v_pref, state.self_state.theta]).to(self.device)
            state_tensors.append(standardize_dim(self_tensor))
                
        # Create dummy if no valid states
        if not state_tensors:
            logging.warning("No valid states in transform method, using zero tensor")
            tensor = torch.zeros(1, self.joint_state_dim, device=self.device)
            return tensor
        
        # Standardize shapes
        state_tensor = torch.stack(state_tensors)
        
        # Apply rotation transformation safely
        try:
            rotated_tensor = self.rotate(state_tensor)
        except Exception as e:
            logging.warning(f"Error in rotation: {e}, using state tensor directly")
            rotated_tensor = state_tensor

        # Add occupancy map if configured
        if self.with_om:
            if hasattr(state, 'human_states') and state.human_states:
                try:
                    occupancy_maps = self.build_occupancy_maps(state.human_states)
                    # Ensure correct dimensions for concatenation
                    if rotated_tensor.shape[0] != occupancy_maps.shape[0]:
                        if rotated_tensor.shape[0] < occupancy_maps.shape[0]:
                            rotated_tensor = rotated_tensor.repeat(occupancy_maps.shape[0], 1)
                        else:
                            occupancy_maps = occupancy_maps.repeat(rotated_tensor.shape[0], 1)
                    rotated_tensor = torch.cat([rotated_tensor, occupancy_maps], dim=1)
                except Exception as e:
                    logging.warning(f"Error adding occupancy map: {e}")
                    # Add dummy occupancy map if error
                    dummy_om = torch.zeros(rotated_tensor.shape[0], self.cell_num ** 2)
                    rotated_tensor = torch.cat([rotated_tensor, dummy_om], dim=1)
        
        # Final check to ensure tensor has the correct dimensions
        if rotated_tensor.shape[1] != self.joint_state_dim:
            logging.debug(f"Adjusting final tensor dimension from {rotated_tensor.shape[1]} to {self.joint_state_dim}")
            rotated_tensor = standardize_dim(rotated_tensor, self.joint_state_dim)
            
        # Ensure we return the correct shape
        if len(rotated_tensor.shape) == 2:
            return rotated_tensor
        else:
            # Reshape if necessary
            return rotated_tensor.reshape(-1, self.joint_state_dim)
    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        This method works for multiple states (batch_size > 1)
        
        Returns: tensor of shape (batch_size, feature_size)
        """
        if state is None:
            return torch.zeros((1, 9), device=self.device)
            
        try:
            # Check for empty input or wrong dimensions
            if state.shape[0] == 0:
                return torch.zeros((0, 9), device=self.device)
                
            # Ensure state is 2D with shape (batch_size, feature_dim)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            elif len(state.shape) > 2:
                # Reshape to 2D, preserving batch dimension
                state = state.reshape(state.shape[0], -1)
            
            # Get features with error handling
            # Extract coordinates safely with bounds checking
            feature_size = state.shape[1]
            
            # Prepare output tensor with safe dimensions
            batch_size = state.shape[0]
            rotated_tensor = torch.zeros((batch_size, 9), device=self.device)
            
            # Process each sample individually to handle dimension errors
            for i in range(batch_size):
                sample = state[i]
                
                try:
                    # Extract position and goal coordinates
                    # For position, use first 2 elements if available
                    if feature_size >= 2:
                        px = sample[0].item()
                        py = sample[1].item()
                    else:
                        px, py = 0, 0
                        
                    # For velocity, use next 2 elements if available
                    if feature_size >= 4:
                        vx = sample[2].item()
                        vy = sample[3].item()
                    else:
                        vx, vy = 0, 0
                        
                    # For radius, use element 4 if available
                    if feature_size >= 5:
                        radius = sample[4].item()
                    else:
                        radius = 0.3  # Default radius
                        
                    # For goal coordinates, use elements 5,6 if available
                    if feature_size >= 7:
                        gx = sample[5].item()
                        gy = sample[6].item()
                    else:
                        gx, gy = 0, 0
                        
                    # For preferred velocity, use element 7 if available
                    if feature_size >= 8:
                        v_pref = sample[7].item()
                    else:
                        v_pref = 1.0  # Default preferred velocity
                        
                    # For theta, use element 8 if available
                    if feature_size >= 9:
                        theta = sample[8].item()
                    else:
                        theta = 0  # Default orientation
                    
                    # Compute relative goal coordinates
                    dx = gx - px
                    dy = gy - py
                    rot = np.arctan2(dy, dx)
                    
                    # Rotate velocity vector
                    vx_rot = vx * np.cos(rot) + vy * np.sin(rot)
                    vy_rot = -vx * np.sin(rot) + vy * np.cos(rot)
                    
                    # Compute distance to goal
                    goal_dist = np.sqrt(dx**2 + dy**2)
                    
                    # Store rotated features in output tensor
                    rotated_tensor[i, 0] = goal_dist  # Goal distance
                    rotated_tensor[i, 1] = vx_rot     # Rotated x velocity
                    rotated_tensor[i, 2] = vy_rot     # Rotated y velocity 
                    rotated_tensor[i, 3] = radius     # Radius
                    rotated_tensor[i, 4] = v_pref     # Preferred velocity
                    rotated_tensor[i, 5] = theta      # Orientation
                    
                    # Extra features if available
                    if feature_size > 9:
                        rotated_tensor[i, 6:9] = torch.tensor([0, 0, 0], device=self.device)
                    
                except Exception as e:
                    logging.warning(f"Error rotating sample {i}: {e}")
                    # Default to zeros if rotation fails
                    rotated_tensor[i] = torch.zeros(9, device=self.device)
            
            return rotated_tensor
            
        except Exception as e:
            logging.error(f"Error in rotate method: {e}")
            # Return safe default tensor
            if isinstance(state, torch.Tensor):
                return torch.zeros((state.shape[0], 9), device=self.device)
            else:
                return torch.zeros((1, 9), device=self.device)

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

