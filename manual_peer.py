import numpy as np

from abc import ABC
from callbacks import PeerEvalCallback

class ManualPeer(ABC):
    def __init__(self, seed=None, **_):
        self.followed_peer = None
        self.n_peers = None
        self.group = None
        self.peer_values = dict()
        self.peer_value_functions = dict()
        self.buffer = None
        self.epoch = 0
        self.policy = self  # for suggestions
        self.np_random = np.random.default_rng(seed)
        self.timesteps = 0

    def learn(self, *_1, total_timesteps=0, callback=None, **_2):

        self.followed_peer = np.where(self.group.peers == self)
        if hasattr(callback, "__getitem__") and\
                type(callback[-1]) == PeerEvalCallback:
            callback[-1].commit_global_step(total_timesteps)

    def _predict(self, obs, **kwargs):
        # needs to be overwritten in subclasses
        raise NotImplementedError

    def predict(self, obs, **kwargs):
        # handle VecEnvs
        actions = [self._predict(o, **kwargs) for o in np.atleast_2d(obs)]
        return np.array(actions), None
