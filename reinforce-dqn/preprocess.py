'''
Author: David Megli
Date: 2025-05-02
File: preprocess.py
Description: Replay Buffer implementation
'''
import cv2
import numpy as np
import gymnasium as gym
from collections import deque

class FrameProcessor:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height

    def __call__(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Add channel dimension
        return resized.astype(np.uint8)

class FrameStacker(gym.Wrapper):
    def __init__(self, env, k=3, width=84, height=84):
        super().__init__(env)
        self.k = k
        self.processor = FrameProcessor(width, height)
        self.frames = deque(maxlen=k)

        # Override observation space to (k, H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, height, width), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self.processor(obs)
        for _ in range(self.k):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self.processor(obs)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info
