import gym
import numpy as np
from .hallway import Hallway


class RoomEnv(Hallway):

    def __init__(self, length=9, dims=1, **kwargs):
        super().__init__(size=[length] * dims, num_actions=2*dims, **kwargs)
        self.start_location = self.size // 2  # start in the middle
        self.observation_space = gym.spaces.Box(0, 1, shape=(2 * dims,),
                                                dtype=np.float64)
        self.np_random = np.random.default_rng()

    def reset(self):
        pos = super().reset()
        # goal location at the boarder
        # one dimension needs to be 0 or size -1
        self.goal_location = self.np_random.integers(0, self.size - 1)
        d = self.np_random.choice(len(self.size))
        self.goal_location[d] = self.np_random.choice([0, self.size[d] - 1])

        return np.hstack((pos, self.goal_location / (self.size - 1)))

    def step(self, action: int):
        pos, reward, done, info = super().step(action)
        obs = np.hstack((pos, self.goal_location / (self.size - 1)))
        return obs, reward, done, info

    def render(self, mode="human"):
        raise NotImplementedError


gym.envs.registration.register(
    id="Room-v21",
    entry_point="env:RoomEnv",
    max_episode_steps=100,
    reward_threshold=0.99,  # 2D: max_steps - (length -1) * 3 / 4
    kwargs=dict(
        length=21,
        dims=2
    )
)

gym.envs.registration.register(
    id="Room-v27",
    entry_point="env:RoomEnv",
    max_episode_steps=150,
    reward_threshold=0.99,  # 2D: max_steps - (length -1) * 3 / 4
    kwargs=dict(
        length=27,
        dims=2
    )
)

gym.envs.registration.register(
    id="Room-v33",
    entry_point="env:RoomEnv",
    max_episode_steps=250,
    reward_threshold=0.99,  # 2D: max_steps - (length -1) * 3 / 4
    kwargs=dict(
        length=33,
        dims=2
    )
)

gym.envs.registration.register(
    id="Room-v3",
    entry_point="env:RoomEnv",
    max_episode_steps=20,
    reward_threshold=0.99,  # 2D: max_steps - (length -1) * 3 / 4
    kwargs=dict(
        length=3,
        dims=2
    )
)

gym.envs.registration.register(
    id="Room-v6",
    entry_point="env:RoomEnv",
    max_episode_steps=50,
    reward_threshold=0.99,  # 2D: max_steps - (length -1) * 3 / 4
    kwargs=dict(
        length=6,
        dims=2
    )
)

