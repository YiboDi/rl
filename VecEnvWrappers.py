from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
normalized_vec_env = VecNormalize(env)

obs = normalized_vec_env.reset()
for _ in range(10):
    action = [normalized_vec_env.action_space.sample()]
    obs, reward, _, _ = normalized_vec_env.step(action)
    print(obs, reward)