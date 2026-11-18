import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import gymnasium as gym
# MP4 Libraries
from IPython.display import HTML
from base64 import b64encode
import cv2
import numpy as np

def show_video(video_path, width=480):
    with open(video_path, "rb") as f:
        video_data = f.read()
    encoded = b64encode(video_data).decode("utf-8")
    html = f'''
        <video width="{width}" controls>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
    '''
    return HTML(html)

def save_agent_video(env, agent, filename="agent_demo.mp4"):
    frames = []
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state

    height, width, layers = frames[0].shape
    out = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    

if "__name__" == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    print(env.action_space.n)
    print(env.observation_space.shape[0])
    class RandomAgent:
        def select_action(self, state):
            return env.action_space.sample()
    save_agent_video(env, RandomAgent(), filename="random_rollout.mp4")
    show_video("random_rollout.mp4")

