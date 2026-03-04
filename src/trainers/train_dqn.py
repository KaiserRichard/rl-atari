"""
DQN Trainer

This script orchestrates the training process.

Responsibilities:
- Load configuration
- Create environment
- Initialize agent
- Run interaction loop
"""

import os
import yaml
import sys 
import torch

# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# Load YAML config
def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)



from utils.device import get_device
from utils.schedule import linear_schedule
from utils.seed import set_seed
from envs.atari_env import make_env
from agents.dqn_agent import DQNAgent

def main(config_path="configs/dqn_breakout.yaml"): 
    # Load configuration
    config = load_config(config_path)
    print(f"Load config from {config_path}")

    # Set up device 
    device = get_device()
    
    # Set up seed
    # If seed is not defined in YAML -> default to 42
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Using device: {device} | seed: {seed}")

    # Create environment
    env_id = config["env_id"]
    env = make_env(env_id=env_id, seed=seed)

    obs, _ = env.reset()
    print(f"Environment: {env_id} created. Action space: {env.action_space.n}")

    # Initialize Agent
    agent = DQNAgent(env=env,config=config,device=device)
    print("DQN Agent initialized successfully")

    # ---------------- Main training loop --------------
    total_timesteps = config["total_timesteps"]
    print("Starting training....")
    
    for step in range(total_timesteps):
        # 1. Select Action (Epsilon-greedy)
        action = agent.select_action(obs, step)

        # 2. Agent interacts with the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. Store transition in Replay Buffer 
        agent.store(obs, next_obs, action, reward, done, info)
        
        # 4. Advance the state
        obs = next_obs

        # If episode ended -> reset environment for the next game
        if done: 
            obs, _ = env.reset()

        # 5. Learning phase (Train every 4 steps after memory is sufficiently full)
        if step > config["learning_starts"] and step % 4 == 0:
            loss = agent.train_step()

        # 6. Target Network Sync
        if step > config["learning_starts"] and step % config["target_update_freq"] == 0: 
            agent.update_target()

        # Print progress
        if step % 10_000 == 0 and step > 0:
            print(f"Step: {step}/{total_timesteps}")

        # Optional: Save a backup checkpoint every 100,000 steps
        if step % 100_000 == 0 and step > 0:
            os.makedirs("outputs/checkpoints", exist_ok=True)
            torch.save(agent.q_net.state_dict(), f"outputs/checkpoints/dqn_breakout_step_{step}.pth")

    env.close()
    print("Training finished.")
    # SAVE THE FINAL MODEL!
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(agent.q_net.state_dict(), "outputs/models/dqn_breakout_final.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()