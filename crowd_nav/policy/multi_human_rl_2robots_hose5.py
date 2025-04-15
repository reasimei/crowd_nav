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


    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        # 添加调试信息
        logging.debug(f"MultiHumanRL.predict called with phase={self.phase}")
        
        # 目标检查 - 如果到达目标，停止机器人
        if self.reach_destination(state):
            self.warning_logged = False
            # 重置上一次动作，这样下次调用时不会使用停止前的动作进行平滑
            if hasattr(self, 'last_action'):
                self.last_action = None
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        
        # 如果接近目标（距离小于特定阈值），降低速度
        goal_distance = np.linalg.norm([state.self_state.px - state.self_state.gx, 
                                       state.self_state.py - state.self_state.gy])
        approaching_goal = goal_distance < 0.5
        
        # 初始化动作空间
        if self.action_space is None:
            try:
                self.build_action_space(state.self_state.v_pref)
                logging.debug(f"Action space built with {len(self.action_space)} actions")
            except Exception as e:
                logging.error(f"Error building action space: {e}")
                return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        occupancy_maps = None
        probability = np.random.random()
        
        # 训练阶段使用ε-贪婪策略
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            
            # 初始化默认动作
            default_action = self.action_space[0]
            
            robot_num = len(state.robot_states) if hasattr(state, 'robot_states') else 8
            actions = []
            
            try:
                # 接近目标时，只考虑低速动作
                filtered_action_space = self.action_space
                if approaching_goal:
                    logging.debug("接近目标，使用低速动作")
                    if self.kinematics == 'holonomic':
                        filtered_action_space = [a for a in self.action_space if 
                                              np.linalg.norm([a.vx, a.vy]) < 0.5 * state.self_state.v_pref]
                    else:
                        filtered_action_space = [a for a in self.action_space if 
                                              a.r < 0.5 * state.self_state.v_pref]
                
                # 确保过滤后的动作空间不为空
                if not filtered_action_space:
                    filtered_action_space = self.action_space
                    
                for action in filtered_action_space:
                    next_self_state = self.propagate(state.self_state, action)
                    
                    for i in range(robot_num):
                        actions.append(action)
                        
                    if self.query_env:
                        next_human_states, reward, done, info = self.env.onestep_lookahead1(actions)
                    else:
                        next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                        for human_state in state.human_states]
                        reward = self.compute_reward(next_self_state, next_human_states)
                    
                    # 安全处理状态批处理
                    try:
                        batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                                    for next_human_state in next_human_states], dim=0)
                    except RuntimeError as e:
                        logging.warning(f"Error in state batching: {e}")
                        continue
                        
                    rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                    
                    if self.with_om:
                        if occupancy_maps is None:
                            occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                        rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                    
                    # 安全处理值计算，并应用动作平滑
                    try:
                        next_state_value = self.model(rotated_batch_input).data.item()
                        value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                        
                        # 添加动作平滑项 - 如果有上一个动作，奖励与上一个动作相似的新动作
                        if hasattr(self, 'last_action') and self.last_action is not None:
                            # 计算与上一个动作的差异
                            if self.kinematics == 'holonomic':
                                action_change = np.linalg.norm([
                                    action.vx - self.last_action.vx, 
                                    action.vy - self.last_action.vy
                                ])
                            else:
                                action_change = abs(action.r - self.last_action.r) + abs(action.v - self.last_action.v)
                            
                            # 平滑系数 - 接近目标时增加平滑度
                            smoothing_factor = 0.1
                            if approaching_goal:
                                smoothing_factor = 0.3
                            
                            # 应用平滑惩罚 - 只针对较大的动作变化
                            if action_change > 0.1:
                                smoothing_penalty = smoothing_factor * action_change
                                value -= smoothing_penalty
                                logging.debug(f"动作平滑惩罚: -{smoothing_penalty:.2f}")
                        
                        self.action_values.append(value)
                        if value > max_value:
                            max_value = value
                            max_action = action
                    except Exception as e:
                        logging.warning(f"Error in value computation: {e}")
                        continue

            except Exception as e:
                logging.error(f"Error in action selection: {e}")
                return default_action

            # 如果没有找到有效动作，使用默认动作
            if max_action is None:
                if not self.warning_logged:
                    logging.warning('No valid action found, using default action')
                    self.warning_logged = True
                max_action = default_action
                
            # 如果接近目标，降低选定动作的速度
            if approaching_goal and max_action is not None:
                if self.kinematics == 'holonomic':
                    speed = np.linalg.norm([max_action.vx, max_action.vy])
                    if speed > 0:
                        # 根据与目标的距离缩放速度
                        speed_scale = min(0.5, goal_distance / 0.5)
                        max_action.vx *= speed_scale
                        max_action.vy *= speed_scale
                else:
                    max_action.v = min(max_action.v, 0.5 * state.self_state.v_pref)

        # 保存选择的动作用于后续平滑
        self.last_action = max_action
        
        if self.phase == 'train':
            self.last_state = self.transform(state)

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

