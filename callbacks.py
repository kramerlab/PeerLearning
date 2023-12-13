from typing import List, Union

import time
import gym
import numpy as np
import wandb
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from peer import PeerGroup


class PeerEvalCallback(EvalCallback):
    """
    Callback to track collective measurements about peers.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use
      ``eval_freq = max(eval_freq // n_envs, 1)``

    :param peer_group: The group of peers
    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the
        callback.
    :param log_path: Path to a folder where the evaluations
        (``evaluations.npz``) will be saved. It will be updated at each
        evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has
        not been wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        peer_group: PeerGroup,
        eval_envs: List[Union[gym.Env, VecEnv]],
        n_samples=100,
        **kwargs
    ):
        self.peer_group = peer_group
        self.eval_envs = eval_envs
        self.n_samples = n_samples

        self.last_logged_matrix = None
        self.follow_matrix = np.zeros((len(peer_group), len(peer_group)))

        self.start_time = time.time()

        super().__init__(**kwargs)

    def _on_step(self) -> bool:
        self.accumulate_followed_peers()  # needs to be done at every step

        # log time for debugging etc.
        self.logger.record("time/time_elapsed",
                           time.time() - self.start_time,
                           exclude="tensorboard")

        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if 'agent_values' in self.peer_group.__dict__:
                self.track_agent_values()
            if 'trust_values' in self.peer_group.peers[0].__dict__:
                self.track_trust_values()
            self.track_followed_agent(self.peer_group.active_peer)

            peer = self.peer_group.active_peer
            eval_values = {
                f"Peer{peer}_0/eval/mean_reward": self.last_mean_reward,
            }
            if peer == len(self.peer_group) - 1:
                eval_values["global_step"] = self.n_calls
                wandb.log(eval_values, commit=True)
            else:
                wandb.log(eval_values, commit=False)
        return True

    def track_agent_values(self):
        n_agents = len(self.peer_group.peers)
        for i in range(n_agents):
            agent_value = self.peer_group.agent_values[i]
            wandb.log({'Peer{}_0/eval/agent_value'.format(i): agent_value},
                      commit=False)
        return True

    def track_trust_values(self):
        peer = self.peer_group.active_peer
        trust_i = self.peer_group.peers[peer].trust_values
        for j, el in np.ndenumerate(trust_i):
            wandb.log({'Peer{}_0/eval/trust_{}'.format(peer, j[0]): el},
                      commit=False)
        return True

    def accumulate_followed_peers(self):
        peer = self.peer_group.active_peer
        followed_peer = self.peer_group.peers[peer].followed_peer
        if followed_peer is not None:
            self.follow_matrix[peer, followed_peer] += 1

    def track_followed_agent(self, active_peer):
        if self.last_logged_matrix is None:
            diff = self.follow_matrix
        else:
            diff = self.follow_matrix - self.last_logged_matrix

        for (followed_peer,), count in np.ndenumerate(
                self.follow_matrix[active_peer]):
            wandb.log({'Peer{}_0/eval/follow_count{}'.format(
                active_peer, followed_peer): count},  commit=False)
            # also log difference
            wandb.log({'Peer{}_0/eval/follow_count_{}diff'.format(
                active_peer, followed_peer): diff[active_peer, followed_peer]},
                      commit=False)
        self.last_logged_matrix = np.copy(self.follow_matrix)

    def commit_global_step(self, timesteps):
        if self.peer_group.active_peer == len(self.peer_group) - 1:
            eval_values = {"global_step": self.n_calls + self.eval_freq}
            wandb.log(eval_values, commit=True)

        self.n_calls += timesteps
