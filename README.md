```markdown
# 🎮 RL-Atari

A modular Deep Reinforcement Learning framework for training agents on Atari environments using PyTorch.

Currently implemented:
- ✅ Deep Q-Network (DQN)

Planned:
- ⏳ Double DQN
- ⏳ Dueling DQN
- ⏳ Advantage Actor-Critic (A2C)

---

## 📁 Project Structure

```text
RL-Atari/
│
├── configs/                  # YAML configuration files
│   └── dqn_breakout.yaml
│
├── src/
│   ├── agents/               # Algorithm implementations
│   │   └── dqn_agent.py
│   │
│   ├── envs/                 # Environment wrappers
│   │   └── atari_env.py
│   │
│   ├── models/               # Neural network architectures
│   │   └── deep_q_network.py
│   │
│   ├── trainers/             # Training scripts
│   │   └── train_dqn.py
│   │
│   ├── utils/                # Utilities (device, seed, schedules)
│   │   ├── device.py
│   │   ├── seed.py
│   │   ├── schedule.py
│   │   └── tensor_utils.py
│   │
│   └── __init__.py
│
├── requirements.txt
└── README.md

```

---

## 🧠 Implemented Algorithm

### 🟢 Deep Q-Network (DQN)

Features:

* Experience Replay Buffer
* Target Network Stabilization
* Epsilon-Greedy Exploration (Linear Decay)
* Atari Preprocessing (SB3 wrappers)
* **Automatic Model Checkpointing**

---

## 🎮 Atari Preprocessing

The environment includes:

* NoopReset
* Frame Skipping (4 frames)
* Episodic Life
* Fire Reset
* Reward Clipping
* Resize to 84x84
* Grayscale
* FrameStack(4)

Final state shape:

```text
(4, 84, 84)

```

---

## ⚙️ Installation

Create virtual environment:

```bash
python -m venv rl-env
source rl-env/bin/activate

```

Install dependencies:

```bash
pip install -r requirements.txt

```

Or manually:

```bash
pip install torch torchvision
pip install stable-baselines3
pip install gymnasium[atari,accept-rom-license]
pip install ale-py
pip install pyyaml

```

---

## 🚀 Running Training

From project root:

```bash
python src/trainers/train_dqn.py

```

---

## 📦 Configuration

Hyperparameters are controlled via YAML:

Example: `configs/dqn_breakout.yaml`

```yaml
env_id: BreakoutNoFrameskip-v4
seed: 42
total_timesteps: 5000000

learning_rate: 0.0001
gamma: 0.99

buffer_size: 250000
batch_size: 32

learning_starts: 80000
target_update_freq: 1000

epsilon:
  start: 1.0
  end: 0.01
  duration: 1000000

```

---

## 🏗 Architecture Philosophy

This project follows clean modular design:

* Separation of environment / model / agent / trainer
* Config-driven experiments
* Utility isolation
* Production-style structure

Designed for:

* Educational clarity
* Research experimentation
* Algorithm extensibility

---

## 📌 Future Extensions

* Double DQN
* Dueling Architecture
* A2C
* PPO
* TensorBoard Logging
* Evaluation pipeline

---

## 🧑‍💻 Author

Built as a structured deep RL learning framework.

---

## ⚠️ Notes

* Virtual environment (`rl-env/`) is excluded from version control.
* This project is optimized for CPU but automatically supports CUDA/MPS if available.
