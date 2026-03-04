'''
DQN Agent
This class encapsulates the entire Deep Q-Network algorithm
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

# ReplayBuffer stores past transitions
from stable_baselines3.common.buffers import ReplayBuffer
# Q-Network architecture
from models.deep_q_network import DeepQNetwork
# Epsilon decay schedule
from utils.schedule import linear_schedule
# Convert Numpy array to torch tensor
from utils.tensor_utils import to_tensor, add_batch_dim

class DQNAgent:
    '''
    Encapsulates full DQN algorithm
    '''
    def __init__(self, env, config, device):
        self.env = env
        self.device = device
        self.config = config

        # Number of possible actions in environment
        self.action_dim = env.action_space.n

        '''
        q_net: 
        - Main network that we train
        target_net: 
        - Frozen network used to compute stable TD targets
        - Without it, Q-learning becomes unstable 
        because Q-network tries to chase its own moving predictions
        '''
        # Pytorch automatically initializes both networks with DIFFERENT random weights
        self.q_net = DeepQNetwork(self.action_dim).to(device)
        self.target_net = DeepQNetwork(self.action_dim).to(device)
        
        # DQN requires the "student" (q_net) and "teacher" (target_net) to start with the same brain
        # We copy q_net's random weights and overwrite target_net so they are identical clones at Step 0
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=config["learning_rate"]
        )

        '''
        ReplayBuffer:
        - Break correlation between samples 
        - Stabilize training
        - Allow off-policy training
        '''
        self.replay_buffer = ReplayBuffer(
            buffer_size=config["buffer_size"],
            observation_space=env.observation_space,
            action_space=env.action_space, 
            device=device,
            optimize_memory_usage=True
        )

    # Action selection
    def select_action(self, obs, step):
        # Compute epsilon using linear decay
        epsilon = linear_schedule(
            start=self.config["epsilon"]["start"],
            end=self.config["epsilon"]["end"],
            duration=self.config["epsilon"]["duration"],
            step=step
        )

        # Exploration
        if random.random() < epsilon:
            return self.env.action_space.sample()

        # Exploitation
        # We are just playing the game right now (not training), we don't need gradients
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = to_tensor(obs, device=self.device)

            # Add batch dimension because Neural Networks expect batch dimension
            if obs_tensor.dim() == 3:
                obs_tensor = add_batch_dim(obs_tensor)

            # Compute Q-values
            q_values = self.q_net(obs_tensor)
            
            # Select action with highest Q-value
            return q_values.argmax(dim=1).item()

    # Store Transition
    def store(self, obs, next_obs, action, reward, done, info):
        '''
        Save experience into replay buffer.
        This allows learning later from past experiences
        '''
        self.replay_buffer.add(obs, next_obs, action, reward, done, info)

    # TRAINING STEP
    def train_step(self):
        '''
        Perform one gradient update
        Steps:
        1. Sample batch from replay buffer
        2. Compute TD target
        3. Compute current Q estimate
        4. Compute Loss
        5. Backpropagate
        '''
        # Sample random mini-batch
        data = self.replay_buffer.sample(self.config["batch_size"])
        
        '''
        Bellman Equation:
            y = r + gamma * max a' Q_targets(s' , a')
        If episode ended:
            no future reward -> multiply by (1-done)
        '''
        # We freeze the Target Network so we don't accidentally train it during backprop
        with torch.no_grad():
            next_q = self.target_net(data.next_observations).max(dim=1)[0]
            td_target = (
                data.rewards.flatten() + self.config["gamma"] * next_q * (1 - data.dones.flatten())
            )
        
        # Use the main DQN to get current predictions
        qsa = self.q_net(data.observations)

        # 1 stands for Dimension 1: Columns
        # squeeze simply removes those extra brackets to make it a flat list
        # before: [[20],[15],[40]] shape: 3,1 -> after: [20, 15, 40] shape:3  
        predicted_rewards = qsa.gather(1, data.actions).squeeze()

        '''
        LOSS
        DQN uses Mean Squared Error between:
            current Q(s,a)
            and TD target
        '''
        loss = F.mse_loss(predicted_rewards, td_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    # Target Network Update
    def update_target(self):
        '''
        Copy weights from q_net to target_net
        This happens periodically to stabilize training
        '''
        self.target_net.load_state_dict(self.q_net.state_dict())