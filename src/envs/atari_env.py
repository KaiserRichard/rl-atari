''''
Atari Environment Creation

Apply:
- Atari Wrappers 
- Preprocess Observations: 
    Resize to 84x84, Grayscale, Frame skipping & Frame stacking
    Final observation shape: (4, 84, 84)
'''

import gymnasium as gym
from stable_baselines3.common.atari_wrappers\
    import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv

def make_env(env_id="BreakoutNoFrameskip-v4", seed=None):
    '''
        Creates Atari environments with all preprocessing wrappers applied
    '''

    # Base environment
    env = gym.make(env_id)

    # Record episodic returns 
    env = gym.wrappers.RecordEpisodeStatistics(env)

    '''
    Atari-specific wrappers
    '''
    # 1. Random number of no-operation actions at reset
    env = NoopResetEnv(env, noop_max=20)

    # 2. Repeat action for 4 frames + max pooling
    env = MaxAndSkipEnv(env, skip=4)

    #3. Treat life loss as episode termination
    env = EpisodicLifeEnv(env)

    #4. Automatically press FIRE if required
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    #5. Clip reward to {-1, 0, 1}
    env = ClipRewardEnv(env)

    '''
    Observation preprocessing
    '''
    env = gym.wrappers.ResizeObservation(env, (84,84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    # For reproducibility
    if seed is not None:
        env.reset(seed=seed)


    return env