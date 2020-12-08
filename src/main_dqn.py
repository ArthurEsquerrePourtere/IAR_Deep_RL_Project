from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from policies.generic_net import GenericNet
from policies.policy_wrapper import PolicyWrapper
from policies.dqn import QNet
import copy

# configurations
env = gym.make("MountainCar-v0")
observe_dim = 2
action_num = 3
max_episodes = 500
max_steps = 200
solved_reward = float('Inf')
solved_repeat = 5
best=-float('Inf')

def distance(pos,savePos):
    #print(savePos)
    CsavePos=savePos.copy()
    CsavePos=[abs((v-pos)) for v in CsavePos]
    CsavePos=sorted(CsavePos)
    rew=sum(CsavePos[0:20]) #pour les 20 plus proches états deja atteints
    return float(rew)


if __name__ == "__main__":
    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)
    save_qnet=copy.deepcopy(q_net)

    dqn = DQN(q_net, q_net_t,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'),learning_rate = 0.002)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    rew=[]
    rew_smoothed=[]

    states=[]

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        dt_reward=0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)


        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise(
                    {"state": old_state}
                )
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)

                total_reward += reward
                or_rew=reward
                reward=0.0

                if(terminal or step > max_steps):
                    states.append(state.numpy()[0,0])
                    if (terminal and step<max_steps):
                        reward=100
                        dt_reward=distance(state.numpy()[0,0],states)
                    else:
                        if len(states)>=20:
                            print(state.numpy()[0,0])
                            reward=distance(state.numpy()[0,0],states)
                            #print(type(reward))
                        else:
                            reward=0.0
                        dt_reward=reward
                

                dqn.store_transition({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })

        

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward)+"    d: "+str(dt_reward))
        rew.append(dt_reward)
        rew_smoothed.append(smoothed_total_reward)

        if(total_reward>best):
            save_qnet=copy.deepcopy(q_net)

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0

    plt.plot(rew)
    plt.savefig('reward_dqn.png')
    plt.close()

    plt.plot(rew_smoothed)
    plt.savefig('reward_smoothed_dqn.png')
    plt.close()

    plt.hist(sorted(states),bins=20)
    plt.savefig('f_states_histo.png')
    plt.close()

    pw = PolicyWrapper(save_qnet, 'dqn', 'MountainCar-v0', 'Arthur_Esquerre-Pourtere', max_steps)
    pw.save(max(rew))