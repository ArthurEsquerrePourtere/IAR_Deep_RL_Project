from machin.frame.algorithms import DDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from policies.generic_net import GenericNet
from policies.policy_wrapper import PolicyWrapper
from policies.ddpg import Actor, Critic
from os import chdir

# configurations
env_name="Pendulum-v0"
env = gym.make(env_name)
observe_dim = 3
action_dim = 1
action_range = 2
max_episodes = 500
max_steps = 200
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = 0
solved_repeat = 5

if __name__ == "__main__":
    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)

    ddpg = DDPG(actor, actor_t, critic, critic_t,
                t.optim.Adam,
                nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    rew=[]
    rew_smoothed=[]

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = ddpg.act_with_noise(
                            {"state": old_state},
                            noise_param=noise_param,
                            mode=noise_mode
                        )
                state, reward, terminal, _ = env.step(action.numpy())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward[0]

                ddpg.store_transition({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward[0],
                    "terminal": terminal or step == max_steps
                })

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                ddpg.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))
        rew.append(total_reward)
        rew_smoothed.append(smoothed_total_reward)

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    # plt.plot(rew)
    # plt.savefig('reward.png')
    # plt.close()

    # plt.plot(rew_smoothed)
    # plt.savefig('reward_smoothed.png')

    pw = PolicyWrapper(actor, 'ddpg', 'Pendulum-v0', 'Arthur_Esquerre-Pourtere', max_steps)
    pw.save(max(rew))
