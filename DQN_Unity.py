import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from collections import deque
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

# Local import
from agent import Agent
from DQN_model import DQN_Fast, DQN_Slow, DQN_ResNet
from stats_side_channel import StatsSideChannel


def get_grid_based_perception(agent_obs):
    state = agent_obs[0]
    grid_based_perception = np.transpose(state, (2, 0, 1))

    return np.expand_dims(grid_based_perception, axis=0)


def dqn_model_trainer(
    env,
    agent,
    n_episodes=100,
    print_range=10,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    early_stop=13,
    verbose=False,
):
    """Deep Q-Learning trainer.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        print_range (int): range to print partials results
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        early_stop (int): Stop training when achieve a defined score respecting 10 min n_episodes.
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=print_range)  # last 100 scores
    # scores_mean = []
    eps = eps_start  # initialize epsilon

    env_object = list(env._env_specs)[0]

    decision_steps, terminal_steps = env.get_steps(env_object)
    agent_ids = decision_steps.agent_id
    agent_id = agent_ids[0]

    for i in range(1, n_episodes + 1):

        env.reset()
        decision_steps, terminal_steps = env.get_steps(env_object)
        agent_ids = decision_steps.agent_id

        score = 0

        agent_obs = decision_steps[agent_id].obs
        state = get_grid_based_perception(agent_obs)

        while True:

            if agent_id in agent_ids:
                act = agent.act(state, eps)
                move_action, laser_action, conf = act

                env.set_action_for_agent(
                    env_object,
                    agent_id,
                    ActionTuple(move_action, laser_action),
                )

            env.step()

            decision_steps, terminal_steps = env.get_steps(env_object)
            agent_ids = decision_steps.agent_id

            if agent_id in agent_ids:
                agent_obs = decision_steps[agent_id].obs
                next_state = get_grid_based_perception(agent_obs)
            else:
                next_state = None

            reward = decision_steps[agent_id].reward

            terminated_agent_ids = terminal_steps.agent_id
            done = (
                terminal_steps[agent_id].interrupted
                if agent_id in terminated_agent_ids
                else False
            )

            action = np.argmax(move_action)
            agent.step(state, action, reward, next_state, done, i)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if verbose:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(i, np.mean(scores_window)),
                end="",
            )
            if i % print_range == 0:
                print(
                    "\rEpisode {}\tAverage Score: {:.2f}".format(
                        i, np.mean(scores_window)
                    )
                )

        if np.mean(scores_window) >= early_stop and i > 10:
            if verbose:
                print(
                    "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                        i, np.mean(scores_window)
                    )
                )
            break

    return scores, i, np.mean(scores_window)


def evaluate_model(env, agent, number_of_episodes=1):

    for threshold in [0.35, 0.36, 0.37, 0.375, 0.38]:

        env.reset()

        env_object = list(env._env_specs)[0]

        decision_steps, terminal_steps = env.get_steps(env_object)
        agent_ids = decision_steps.agent_id

        # TODO: Get the number of agents from the env
        number_of_fast_slow = len(agent)
        number_of_hybrid = 1
        number_of_random = 1

        number_of_agents = number_of_fast_slow + number_of_hybrid + number_of_random

        all_scores_for_agents = []

        random_actions = []
        hybrid_actions = []

        for run in range(1, number_of_episodes + 1):
            print(f"{run}/{number_of_episodes}")
            scores_for_agents = [0] * number_of_agents
            eps = 0.0

            number_of_fast_actions = 0
            number_of_slow_actions = 0

            number_of_random_fast_actions = 0
            number_of_random_slow_actions = 0

            env.reset()

            agents_interrupted = [False] * number_of_agents

            while not all(agents_interrupted):
                for agent_id in agent_ids:
                    agent_obs = decision_steps[agent_id].obs
                    state = get_grid_based_perception(agent_obs)

                    if agent_id == 2:
                        act_fast = agent[0].act(state, eps)
                        move_action, laser_action, conf_fast = act_fast

                        if conf_fast < threshold:
                            act_slow = agent[-1].act(state, eps)
                            move_action, laser_action, conf_slow = act_slow
                            number_of_slow_actions += 1
                        else:
                            number_of_fast_actions += 1
                    elif agent_id == 3:
                        if np.random.randint(2, size=1) == 1:
                            act_fast = agent[0].act(state, eps)
                            move_action, laser_action, conf_fast = act_fast
                            number_of_random_fast_actions += 1
                        else:
                            act_slow = agent[-1].act(state, eps)
                            move_action, laser_action, conf_slow = act_slow
                            number_of_random_slow_actions += 1
                    else:
                        act = agent[agent_id].act(state, eps)
                        move_action, laser_action, _ = act

                    env.set_action_for_agent(
                        env_object,
                        agent_id,
                        ActionTuple(move_action, laser_action),
                    )
                    reward = decision_steps[agent_id].reward
                    terminated_agent_ids = terminal_steps.agent_id
                    done = (
                        terminal_steps[agent_id].interrupted
                        if agent_id in terminated_agent_ids
                        else False
                    )
                    scores_for_agents[agent_id] += reward

                    if done:
                        agents_interrupted[agent_id] = True

                env.step()
                decision_steps, terminal_steps = env.get_steps(env_object)
                agent_ids = decision_steps.agent_id

            print(f"{run}: Scores: {scores_for_agents}")
            print(f"Hybrid: <{number_of_fast_actions}, {number_of_slow_actions}>")
            print(
                f"Random: <{number_of_random_fast_actions}, {number_of_random_slow_actions}>"
            )

            random_actions.append(
                [number_of_random_fast_actions, number_of_random_slow_actions]
            )
            hybrid_actions.append([number_of_fast_actions, number_of_slow_actions])

            all_scores_for_agents.append(scores_for_agents)

        with open(
            f"./results_unity/{threshold}_hybrid_selection_distribution.json", "w"
        ) as hybrid_network_selection_distribution:
            json.dump(hybrid_actions, hybrid_network_selection_distribution, indent=2)

        with open(
            f"./results_unity/{threshold}_random_network_selection_distribution.json",
            "w",
        ) as random_network_selection_distribution:
            json.dump(random_actions, random_network_selection_distribution, indent=2)

        with open(
            f"./results_unity/{threshold}_all_scores_for_agents.json", "w"
        ) as score_file:
            json.dump(all_scores_for_agents, score_file, indent=2)

        plt.figure()
        plt.title(f"Mean reward over {number_of_episodes} runs")
        plt.ylabel("Reward")
        name = ["Fast", "Slow", "Hybrid", "Random"]
        plt.bar(name, np.mean(all_scores_for_agents, axis=0))
        plt.savefig(f"results_unity/{threshold}_{number_of_episodes}_evaluations.png")


def main(args):
    SEED = 10
    FRAMES = 1
    LEARNING_RATE = 5e-4
    N_EPISODES = 500
    BENCHMARK_MEAN_REWARD = 30
    NUM_EPISODES = 75
    VERBOSE = True

    ENV_NAME = "FoodCollector"

    TRAIN_MODEL = args.train
    ENV_PATH = "/Users/hermannordaunet/Documents/UiO/Semester9/IN5490/IN5490-project/unity_builds/FoodCollector"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    side_channel_1 = StatsSideChannel()
    env = UnityEnvironment(
        file_name=ENV_PATH,
        seed=SEED,
        side_channels=[side_channel_1],
        no_graphics=True,
    )
    env.reset()

    behavior_specs = env.behavior_specs
    env_object = list(env._env_specs)[0]
    action_spec = behavior_specs[env_object].action_spec
    observation_spec = behavior_specs[env_object].observation_specs

    continuous_size = action_spec.continuous_size
    discrete_size = action_spec.discrete_size

    input_size = observation_spec[0].shape
    screen_height, screen_width, channels = input_size

    nn_inputs = channels * FRAMES

    if TRAIN_MODEL:

        if args.fast:
            qnetwork_local = DQN_Fast(
                nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
            ).to(DEVICE)
            qnetwork_target = DQN_Fast(
                nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
            ).to(DEVICE)
        elif args.slow:
            qnetwork_local = DQN_ResNet(
                nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
            ).to(DEVICE)
            qnetwork_target = DQN_ResNet(
                nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
            ).to(DEVICE)

        qnetwork_local.train()

        agent = Agent(
            qnetwork_local,
            qnetwork_target,
            seed=SEED,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
        )

        scores, episodes, last_avg_score = dqn_model_trainer(
            env,
            agent,
            n_episodes=N_EPISODES,
            early_stop=BENCHMARK_MEAN_REWARD,
            verbose=VERBOSE,
        )

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.title(ENV_NAME)
        plt.ylabel("Average Score")
        plt.xlabel("Episode #")
        plt.show(block=False)

        # Save model
        if isinstance(agent.qnetwork_local, DQN_ResNet):
            model_path = "./unity_models/model_resnet.pt"
        elif isinstance(agent.qnetwork_local, DQN_Fast):
            model_path = "./unity_models/model.pt"
        elif isinstance(agent.qnetwork_local, DQN_Slow):
            model_path = "./unity_models/model_slow.pt"

        agent.save_model(model_path)

    else:
        qnetwork_local_resnet = DQN_ResNet(
            nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
        ).to(DEVICE)
        qnetwork_target_resnet = DQN_ResNet(
            nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
        ).to(DEVICE)

        qnetwork_local_fast = DQN_Fast(
            nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
        ).to(DEVICE)
        qnetwork_target_fast = DQN_Fast(
            nn_inputs, screen_height, screen_width, continuous_size, None, DEVICE
        ).to(DEVICE)

        number_of_agents = 2
        agent_list = [None] * number_of_agents
        fast_model_path = "./unity_models/model_fast.pt"
        slow_model_path = "./unity_models/model_resnet.pt"

        for agent_id in range(int(number_of_agents / 2)):
            agent_list[agent_id] = Agent(
                qnetwork_local_fast,
                qnetwork_target_fast,
                seed=SEED,
                learning_rate=LEARNING_RATE,
                device=DEVICE,
            )

            agent_list[agent_id].load_model(fast_model_path)

        for agent_id in range(int(number_of_agents / 2), number_of_agents):
            agent_list[agent_id] = Agent(
                qnetwork_local_resnet,
                qnetwork_target_resnet,
                seed=SEED,
                learning_rate=LEARNING_RATE,
                device=DEVICE,
            )
            agent_list[agent_id].load_model(slow_model_path)

        qnetwork_local_resnet.eval()
        qnetwork_target_resnet.eval()

        qnetwork_local_fast.eval()
        qnetwork_target_fast.eval()

        evaluate_model(env, agent_list, number_of_episodes=NUM_EPISODES)

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--train", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true")
    group.add_argument("--slow", action="store_true")

    train_group.add_argument_group(group)

    args = parser.parse_args()
    main(args)
