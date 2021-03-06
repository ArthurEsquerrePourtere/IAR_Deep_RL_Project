import gym


class PendulumWrapper(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the pendulum environment
    """
    def __init__(self, env):
        super(PendulumWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, y = self.env.step(action)
        #reward=(16.2736044+reward)/20   
        return next_state, reward, done, y
