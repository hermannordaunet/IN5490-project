import os
import pygame
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import cv2
from collections import deque

# Local import
from DQN_model import DQN_Fast
from DQN_model import DQN_Slow
from score_logger import ScoreLogger
from replay_memory import ReplayMemory
from DQN import *


############ HYPERPARAMETERS ##############
BATCH_SIZE = 128  # original = 128
GAMMA = 0.99  # original = 0.999
EPS_START = 0.9  # original = 0.9
EPS_END = 0.01  # original = 0.05
EPS_DECAY = 3000  # original = 200
TARGET_UPDATE = 50  # original = 10
MEMORY_SIZE = 100000  # original = 10000
END_SCORE = 400  # 200 for Cartpole-v0  #400
TRAINING_STOP = 400  # threshold for training stop #400
N_EPISODES = 10  # total episodes to be run
LAST_EPISODES_NUM = 20  # number of episodes for calculating mean
EPISODES_AFTER_TRAINING = 10  # number of episodes ran after training is done.
FRAMES = 2  # state is the number of last frames: the more frames,
# the more the state is detailed (still Markovian)
RESIZE_PIXELS = 60  # Downsample image to this number of pixels

THRESHOLD = 0.56


GRAYSCALE = True  # False is RGB
LOAD_MODEL = False  # If we want to load the model, Default= False
USE_CUDA = True  # If we want to use GPU (powerful one needed!)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "CartPole-v1"
############################################


def make_video(images):
    size = 60, 135
    fps = 25
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    img = np.array(images)
    for img in img:
        img = img[0,0,:,:]
    out.write(img)
    out.release()
    


def evaluate(fast_path, slow_path, runs):
    # Settings for GRAYSCALE / RGB
    if GRAYSCALE == 0:
        resize = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                T.ToTensor(),
            ]
        )

        nn_inputs = 3 * FRAMES  # number of channels for the nn
    else:
        resize = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                T.Grayscale(),
                T.ToTensor(),
            ]
        )
        nn_inputs = FRAMES  # number of channels for the nn

    # If gpu is to be used
    if gym.__version__ < "0.26":
        env = gym.make(
            ENV_NAME, new_step_api=True, render_mode="single_rgb_array"
        ).unwrapped
    else:
        env = gym.make(ENV_NAME, render_mode="rgb_array").unwrapped

    # MAIN LOOP
    env.reset()
    init_screen = get_screen(env, resize)
    _, _, screen_height, screen_width = init_screen.shape
    print("Screen height: ", screen_height, " | Width: ", screen_width)
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    scorelogger = ScoreLogger(ENV_NAME, "eval"+slow_path)
    
    total_steps = 0
    for j in range(runs):
        memory = ReplayMemory(MEMORY_SIZE)

        model_fast = DQN_Fast(
                nn_inputs, screen_height, screen_width, n_actions, memory, DEVICE
            ).to(DEVICE)
        model_fast.load_state_dict(torch.load(fast_path, map_location=torch.device('cpu')))
        model_fast.eval()

        model_slow = DQN_Slow(
                nn_inputs, screen_height, screen_width, n_actions, memory, DEVICE
            ).to(DEVICE)
        model_slow.load_state_dict(torch.load(slow_path, map_location=torch.device('cpu')))
        model_slow.eval()

        steps_done = 0
        slow_count = 0
        
        for i_episode in range(N_EPISODES):
            # Initialize the environment and state
            env.reset()
            screens_for_video = []
            init_screen = get_screen(env, resize)
            screens = deque([init_screen] * FRAMES, FRAMES)
            state = torch.cat(list(screens), dim=1)


            for t in count():
                #Select and perform an action
                action, steps_done, conf_fast = select_action(
                    model_fast, state, steps_done, True
                )

                

                if conf_fast < THRESHOLD:
                    action, _, conf_slow = select_action(
                        model_slow, state, steps_done, True
                    )
            
                
                if conf_fast < conf_slow:
                    slow_count += 1

                state_variables, _, done, _, _ = env.step(action.item())
                env.render()

                # Observe new state
                screen = get_screen(env, resize)
                screens.append(screen)
                screens_for_video.append(screen)
                next_state = torch.cat(list(screens), dim=1) if not done else None

                # Reward modification for better stability
                # HERMAN: They calculate reward from the vector observation and not just -1 and 1
                # We might need to do this aswell
                x, x_dot, theta, theta_dot = state_variables
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (
                    env.theta_threshold_radians - abs(theta)
                ) / env.theta_threshold_radians - 0.5
                reward = r1 + r2
                reward = torch.tensor([reward], device=DEVICE)
                if t >= END_SCORE - 1:
                    reward = reward + 20
                    done = 1
                else:
                    if done:
                        reward = reward - 20
                        break
        

                memory.push(state, action, next_state, reward)

                state = next_state
    
    
        print(f"For at total of {steps_done} runs, Slow is more confident {slow_count/steps_done*100}% of the times")
        print(f"run nr:{j}, score: {steps_done}")
        total_steps += steps_done
    env.close()

    print(f"average score over {j} runs is {total_steps/runs}")


