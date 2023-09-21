import os
import pygame
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import deque

# Local import
from DQN_model import DQN_Slow, DQN_ResNet, DQN_Fast
from score_logger import ScoreLogger
from replay_memory import ReplayMemory
from eval import *


############ HYPERPARAMETERS ##############
BATCH_SIZE = 128  # original = 128
GAMMA = 0.99  # original = 0.999
EPS_START = 0.9  # original = 0.9
EPS_END = 0.01  # original = 0.05
EPS_DECAY = 3000  # original = 200
TARGET_UPDATE = 10  # original = 10
MEMORY_SIZE = 100000  # original = 10000
END_SCORE = 3400  # 200 for Cartpole-v0  #400
TRAINING_STOP = 1000  # threshold for training stop #400
N_EPISODES = 1250  # total episodes to be run
LAST_EPISODES_NUM = 20  # number of episodes for calculating mean
EPISODES_AFTER_TRAINING = 10  # number of episodes ran after training is done.
FRAMES = 2  # state is the number of last frames: the more frames,
# the more the state is detailed (still Markovian)
RESIZE_PIXELS = 60  # Downsample image to this number of pixels
CHECKPOINT_INTERVAL = 20


GRAYSCALE = True  # False is RGB
LOAD_MODEL = False  # If we want to load the model, Default= False
USE_CUDA = True  # If we want to use GPU (powerful one needed!)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "CartPole-v1"
############################################

# --------------------------------------------Input extraction------------------------------------------------

# Cart location for centering image crop
def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


# Cropping, downsampling (and Grayscaling) image
def get_screen(env, resize):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(
            cart_location - view_width // 2, cart_location + view_width // 2
        )
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(DEVICE)


# Action selection , if stop training == True, only exploitation
def select_action(model, state, steps_done, stop_training):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            pred, conf = model(state)
            return pred.max(1)[1].view(1, 1), steps_done, conf
    else:
        return (
            torch.tensor(
                [[random.randrange(model.outputs)]],
                device=model.device,
                dtype=torch.long,
            ),
            steps_done,
            1,
        )


episode_durations = []



# ------------------------------------------TRAINING LOOP--------------------------------------------------


# Training
def optimize_model(model, target_model, optimizer):
    if len(model.memory) < BATCH_SIZE:
        return
    transitions = model.memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = model.memory.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=model.device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # torch.cat concatenates tensor sequence
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(model.device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    pred, conf = model(state_batch)

    state_action_values = pred.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=model.device)
    pred, conf = target_model(non_final_next_states)
    next_state_values[non_final_mask] = pred.max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    # plt.figure(2)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def run(runs, episodes_trajectories, fast, path):

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

    stop_training = False

    memory = ReplayMemory(MEMORY_SIZE)
    # If gpu is to be used

    if gym.__version__ < "0.26":
        env = gym.make(
            ENV_NAME, new_step_api=True, render_mode="single_rgb_array"
        ).unwrapped
    else:
        env = gym.make(ENV_NAME, render_mode="rgb_array").unwrapped

    # MAIN LOOP
    stop_training = False
    steps_done = 0

    env.reset()
    init_screen = get_screen(env, resize)
    _, _, screen_height, screen_width = init_screen.shape
    print("Screen height: ", screen_height, " | Width: ", screen_width)
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    scorelogger = ScoreLogger(ENV_NAME, path)

    best_model = [None, 0]

    mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)

    memory = ReplayMemory(MEMORY_SIZE)

    if fast:
        policy_net = DQN_Fast(
            nn_inputs, screen_height, screen_width, n_actions, memory, DEVICE
        ).to(DEVICE)
        target_net = DQN_Fast(
            nn_inputs, screen_height, screen_width, n_actions, memory, DEVICE
        ).to(DEVICE)
    else:
        policy_net = DQN_Slow(
            nn_inputs, screen_height, screen_width, n_actions, memory, DEVICE
        ).to(DEVICE)
        target_net = DQN_Slow(
            nn_inputs, screen_height, screen_width, n_actions, memory, DEVICE
        ).to(DEVICE)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    policy_net.train()

    if fast:
        optimizer = optim.RMSprop(policy_net.parameters())
    else:
        optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025)

    count_final = 0

    steps_done = 0
    episode_durations = []
    for i_episode in range(N_EPISODES):
        # Initialize the environment and state
        env.reset()
        init_screen = get_screen(env, resize)
        screens = deque([init_screen] * FRAMES, FRAMES)
        state = torch.cat(list(screens), dim=1)

        if i_episode % CHECKPOINT_INTERVAL == 0:
            print("checkpoint saved")
            torch.save(
                {
                    "epoch": i_episode,
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "./model/checkpoint/model.pt",
            )

        for t in count():

            # Select and perform an action
            action, steps_done, conf = select_action(
                policy_net, state, steps_done, stop_training
            )
            state_variables, _, done, _, _ = env.step(action.item())

            # Observe new state
            screens.append(get_screen(env, resize))
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

            # Store the transition in memory
            policy_net.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if done:
                episode_durations.append(t + 1)
                mean_last.append(t + 1)
                mean = 0
                scorelogger.add_score(t, i_episode)
                for i in range(LAST_EPISODES_NUM):
                    mean += mean_last[i]
                mean = mean / LAST_EPISODES_NUM
                print(f"Mean of last {LAST_EPISODES_NUM} episode: {mean}")
                if mean > best_model[1]:
                    best_model = [policy_net, mean]
                if mean < TRAINING_STOP and stop_training == False:
                    optimize_model(policy_net, target_net, optimizer)
                else:
                    stop_training = True
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if stop_training == True:
            count_final += 1
            if count_final >= EPISODES_AFTER_TRAINING:
                break

    print("Complete")
    stop_training = False
    episodes_trajectories.append(episode_durations)

    return best_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--eval", default=False)
    parser.add_argument("--fast_path")
    parser.add_argument("--slow_path")
    parser.add_argument("--runs", default=1)

    args = parser.parse_args()

    episodes_trajectories = []

    if not args.eval:
        if args.fast_path:
            fast_model = run(args.runs, episodes_trajectories, True, args.fast_path)[0]
            torch.save(fast_model.state_dict(), args.fast_path)
        if args.slow_path:
            slow_model = run(args.runs, episodes_trajectories, False, args.slow_path)[0]
            torch.save(slow_model.state_dict(), args.slow_path)

    elif args.fast_path is not None and args.slow_path is not None:
        evaluate(args.fast_path, args.slow_path, int((args.runs)))
    else:
        print("Need both model paths")
