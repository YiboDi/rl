# import gymnasium as gym
# from stable_baselines3 import A2C, SAC, PPO, TD3

# import os

# # Create save dir
# save_dir = "/tmp/gym/"
# os.makedirs(save_dir, exist_ok=True)

# model = PPO("MlpPolicy", "Pendulum-v1", verbose=0).learn(8_000)
# # The model will be saved under PPO_tutorial.zip
# model.save(f"{save_dir}/PPO_tutorial")
# print(f"{save_dir}/PPO_tutorial" )

# # sample an observation from the environment
# obs = model.env.observation_space.sample()

# # Check prediction before saving
# print("pre saved", model.predict(obs, deterministic=True))

# del model  # delete trained model to demonstrate loading

# loaded_model = PPO.load(f"{save_dir}/PPO_tutorial")
# # Check that the prediction is the same after loading (for the same observation)
# print("loaded", loaded_model.predict(obs, deterministic=True))

import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# # Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1)
# # Train the agent and display a progress bar
# model.learn(total_timesteps=int(2e5), progress_bar=True)
# # Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")