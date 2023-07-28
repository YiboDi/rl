import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander", render_mode = 'human')

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=int(1e5), progress_bar=True)

model.save("dqn_lunarlander")
del model

model = DQN.load("dqn_lunarlander", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("rgb-array")

