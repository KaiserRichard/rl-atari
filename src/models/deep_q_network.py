''''
Q network for Atari DQN
'''
import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __int__(self, action_dims):
        super().__init__()

        self.network = nn.Sequential(
            # Input (4, 84, 84)

            # Conv 1
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Conv2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Conv3
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),

            # Fully connected
            nn.Linear(3136, 512),
            nn.ReLU(),


            #Output Q-values
            nn.Linear(512, action_dims)
        )
        
        def forward(sefl, x):
            '''
            Normalize  pixel values to [0. 1] by dividing by 255
            '''
            return self.network(x / 255.0)
