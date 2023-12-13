import gym
import numpy as np


class Hallway(gym.Env):

    def __init__(self, size=17, num_actions: int = 2, terminate_on_goal=False):
        super().__init__()
        self.size = np.atleast_1d(size)  # enable multiple dimensions
        assert len(self.size) > 0 and (self.size > 0).all and num_actions > 0

        self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.size),))
        self.action_space = gym.spaces.Discrete(num_actions)
        self.agent_location = np.zeros_like(size)
        self.start_location = self.size // 2  # start in the middle

        if self.size[0] < 5:
            self.start_location[0] = 0
        else:
            self.start_location[0] = self.size[0] // 2 - 2  # 2 off the middle

        self.goal_location = self.size // 2
        self.goal_location[0] = self.size[0] - 1
        self.terminate_on_goal = terminate_on_goal

    def reset(self):
        self.agent_location = np.copy(self.start_location)
        return self.agent_location / (self.size - 1)

    def step(self, action: int):
        assert self.action_space.contains(action), f"{action} is not a valid" \
                                                   f"action"
        reward = -0.001
        done = False
        d = action // 2
        direction = action % 2
        if d < len(self.size):  # move in corresponding dimension
            if direction == 0 and self.agent_location[d] < self.size[d] - 1:
                self.agent_location[d] += 1
            if direction == 1 and self.agent_location[d] > 0:
                self.agent_location[d] -= 1
        if np.array_equal(self.agent_location, self.goal_location):
            reward = 1.0
            done = self.terminate_on_goal

        return self.agent_location / (self.size - 1), reward, done, {}

    def render(self, mode="human"):
        raise NotImplementedError


gym.envs.registration.register(
    id="Hallway-v0",
    entry_point="env:Hallway",
    max_episode_steps=200,
    reward_threshold=0.99,
)

gym.envs.registration.register(
    id="Hallway-v1",
    entry_point="env:Hallway",
    max_episode_steps=300,
    reward_threshold=0.99,
    kwargs=dict(
        num_actions=6,
        size=[17, 7]
    )
)
