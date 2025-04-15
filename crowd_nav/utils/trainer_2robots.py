import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


class NetworkBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_params = {}

    def store_params(self, **params):
        """Store initialization parameters for cloning"""
        self.network_params = params

    def clone(self):
        """Create a clone of the network with same parameters"""
        if not self.network_params:
            raise ValueError("Network parameters not stored. Call store_params first.")
            
        clone = type(self)(**self.network_params)
        clone.load_state_dict(self.state_dict())
        return clone


class ValueNetwork(NetworkBase):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, 
                 attention_dims, with_global_state, cell_size, cell_num):
        super().__init__()
        
        # Store init params for cloning
        self.store_params(
            input_dim=input_dim,
            self_state_dim=self_state_dim, 
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            mlp3_dims=mlp3_dims,
            attention_dims=attention_dims,
            with_global_state=with_global_state,
            cell_size=cell_size,
            cell_num=cell_num
        )
        
        # Network architecture parameters
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.attention_dims = attention_dims
        self.with_global_state = with_global_state
        self.cell_size = cell_size
        self.cell_num = cell_num
        
        # MLP layers
        mlp1_dims = [input_dim] + mlp1_dims
        mlp2_dims = [mlp1_dims[-1]] + mlp2_dims
        mlp3_dims = [mlp2_dims[-1]] + mlp3_dims
        
        self.mlp1 = nn.Sequential()
        for i in range(len(mlp1_dims)-1):
            self.mlp1.add_module('fc{}'.format(i), nn.Linear(mlp1_dims[i], mlp1_dims[i+1]))
            self.mlp1.add_module('relu{}'.format(i), nn.ReLU())
            
        self.mlp2 = nn.Sequential()
        for i in range(len(mlp2_dims)-1):
            self.mlp2.add_module('fc{}'.format(i), nn.Linear(mlp2_dims[i], mlp2_dims[i+1]))
            self.mlp2.add_module('relu{}'.format(i), nn.ReLU())
            
        self.mlp3 = nn.Sequential()
        for i in range(len(mlp3_dims)-1):
            self.mlp3.add_module('fc{}'.format(i), nn.Linear(mlp3_dims[i], mlp3_dims[i+1]))
            if i != len(mlp3_dims)-2:
                self.mlp3.add_module('relu{}'.format(i), nn.ReLU())
                
        # Attention mechanism
        self.attention = True if self.attention_dims is not None else False
        if self.attention:
            self.attention_weights = nn.Parameter(torch.randn(self.attention_dims, 1))

    def forward(self, state):
        """
        Forward pass of value network
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        
        if self.attention:
            global_state = torch.mean(mlp2_output.view(size[0], size[1], -1), dim=1)
            attention_weights = torch.matmul(mlp2_output.view(size[0], size[1], -1), self.attention_weights)
            attention_weights = torch.softmax(attention_weights.squeeze(2), dim=1).unsqueeze(2)
            weighted_feature = mlp2_output.view(size[0], size[1], -1) * attention_weights
            global_state = torch.sum(weighted_feature, dim=1)
        else:
            global_state = torch.mean(mlp2_output.view(size[0], size[1], -1), dim=1)
            
        value = self.mlp3(global_state)
        return value


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.batch_size = batch_size
        self.optimizer = None
        self.writer = SummaryWriter(log_dir='runs/training')
        self.data_loader = None
        
        # Initialize target network
        try:
            self.target_model = self.model.clone().to(device)
        except Exception as e:
            logging.error(f"Failed to initialize target network: {e}")
            self.target_model = None
            
        self.train_steps = 0
        self.best_loss = float('inf')
        self.patience = 20
        self.patience_counter = 0
        self.grad_clip = 1.0
        self.target_update_freq = 100
        self.warmup_steps = 1000

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min(1., float(steps) / self.warmup_steps)
        )

    def update_target_network(self, tau=0.01):
        """Soft update target network"""
        try:
            if self.target_model is None:
                self.target_model = self.model.clone().to(self.device)
            else:
                for target_param, local_param in zip(self.target_model.parameters(), 
                                                   self.model.parameters()):
                    target_param.data.copy_(tau * local_param.data + 
                                          (1.0 - tau) * target_param.data)
        except Exception as e:
            logging.error(f"Failed to update target network: {e}")
            raise

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
            
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs).to(self.device)
                values = Variable(values).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.data.item()
                
                self.train_steps += 1
                if self.train_steps % self.target_update_freq == 0:
                    self.update_target_network()

            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('Loss/train', average_epoch_loss, epoch)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)
            
            # Early stopping check
            if average_epoch_loss < self.best_loss:
                self.best_loss = average_epoch_loss
                self.patience_counter = 0
                self.save_model('best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logging.info('Early stopping triggered')
                    break

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
            
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self.data_loader))
            inputs = Variable(inputs).to(self.device)
            values = Variable(values).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.data.item()
            
            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.update_target_network()

        average_loss = losses / num_batches
        self.writer.add_scalar('Loss/batch', average_loss, self.train_steps)
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss

    def save_model(self, filename):
        """Save model checkpoint"""
        path = os.path.join('saved_models', filename)
        os.makedirs('saved_models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'train_steps': self.train_steps
        }, path)

    def cleanup(self):
        """Cleanup resources"""
        self.writer.close()