import argparse
import numpy as np

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment


parser = argparse.ArgumentParser()

parser.add_argument("--n", type=int, default=100)

args = parser.parse_args()


def main():

    env = UnityEnvironment(seed=4, side_channels=[], no_graphics=False)
    env.reset()

    # The environment
    behavior_specs = env.behavior_specs
    env_object = list(env._env_specs)[0]

    action_spec = behavior_specs[env_object].action_spec
    observation_spec = behavior_specs[env_object].observation_specs

    continous_size = action_spec.continuous_size
    discrete_size = action_spec.discrete_size

    n_actions = continous_size + discrete_size

    input_size = observation_spec[0].shape

    for i in range(args.n):

        decision_steps, terminal_steps = env.get_steps(env_object)
        agent_ids = decision_steps.agent_id

        for agent_id in agent_ids:
            agent_obs = decision_steps[agent_id].obs
            # print(f"Agent ID {agent_id}: {agent_obs[0][1][:][:]})")

            # continuous_action = np.random.rand(1, 3)
            continuous_action = np.array([[0.8, 0.1, 0.1]])
            discrete_action = np.random.randint(2, size=(1, 1))

            env.set_action_for_agent(
                env_object,
                agent_id,
                ActionTuple(continuous_action, discrete_action),
            )

            env.step()
            decision_steps, terminal_steps = env.get_steps(env_object)
            next_obs = decision_steps[agent_id].obs

            print(np.array_equal(next_obs, agent_obs))

    env.close()


if __name__ == "__main__":

    main()
