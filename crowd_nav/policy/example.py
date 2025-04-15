import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------------------------
# 模拟的ValueNetwork，仅供示意。实际请参考sarl.py中的ValueNetwork实现。
# 此处假设网络使用LSTM捕捉100个时间步内的时序信息，然后输出最终的价值评估。
# ---------------------------------------------------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super(ValueNetwork, self).__init__()
        # 定义LSTM层：输入维度为input_dim，隐藏层维度hidden_dim，num_layers层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 最后经过全连接层，输出一个标量
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x的形状：[batch, time_steps, input_dim]
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出作为序列的特征表示
        last_time_step = lstm_out[:, -1, :]  # 形状：[batch, hidden_dim]
        value = self.fc(last_time_step)       # 形状：[batch, 1]
        value = value.squeeze(-1)             # 压缩为[batch]，方便计算MSELoss
        return value

# ---------------------------------------------------------------------------
# 参数设置
# ---------------------------------------------------------------------------
num_robots = 8      # 8个机器人
time_steps = 100    # 每个机器人100个时间步
input_dim = 1       # 每个时间步的特征维度，此处为1（例如仅表示速度）

# ---------------------------------------------------------------------------
# 假设LLM已经生成了每个机器人的完整速度序列
# 这里我们用随机数据作为示例，实际中应由LLM生成
# action_sequences的形状：[num_robots, time_steps, input_dim]
# ---------------------------------------------------------------------------
action_sequences = np.random.rand(num_robots, time_steps, input_dim)

# ---------------------------------------------------------------------------
# 假设每个机器人在这个episode的累积奖励由环境计算得到
# 这里同样使用随机数作为示例，实际中请使用真实的累积奖励值
# rewards的形状：[num_robots]
# ---------------------------------------------------------------------------
rewards = np.random.rand(num_robots)

# ---------------------------------------------------------------------------
# 将数据转换为Tensor，并指定计算设备
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_sequences_tensor = torch.tensor(action_sequences, dtype=torch.float32).to(device)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)

# ---------------------------------------------------------------------------
# 实例化价值网络
# ---------------------------------------------------------------------------
value_net = ValueNetwork(input_dim=input_dim, hidden_dim=64, num_layers=1).to(device)

# 使用Adam优化器和均方误差损失函数
optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---------------------------------------------------------------------------
# 前向传播：使用完整序列作为输入，计算网络的预测价值
# ---------------------------------------------------------------------------
predicted_values = value_net(action_sequences_tensor)  # 形状为 [num_robots]
loss = loss_fn(predicted_values, rewards_tensor)

# ---------------------------------------------------------------------------
# 反向传播和参数更新（一次性使用整个episode的数据更新价值网络）
# ---------------------------------------------------------------------------
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("更新后的Loss:", loss.item())
