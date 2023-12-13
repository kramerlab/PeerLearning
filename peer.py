from abc import ABC
from typing import Type
import itertools as it

import numpy as np
import torch

from suggestionbuffer import SuggestionBuffer
from utils import make_env

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class PeerGroup:
    """ A group of peers who train together. """
    def __init__(self, peers, use_agent_values=False, init_agent_values=200.,
                 lr=0.95, switch_ratio=0, use_advantage=False,
                 max_peer_epochs=1_000_000_000):
        """
        :param peers: An iterable of peer agents
        :param lr: The learning rate for trust and agent values
        :param switch_ratio: switch_ratio == 0 means no switching
        :param use_advantage: use advantage instead of value for AV updates
        """
        self.peers = peers
        self.lr = lr
        self.switch_ratio = switch_ratio
        self.active_peer = None  # index of currently learning peer
        self.solo_epoch = False
        self.use_advantage = use_advantage
        self.max_peer_epochs = max_peer_epochs

        if use_agent_values:
            self.agent_values = np.full(len(peers), init_agent_values,
                                        dtype=np.float32)
            key = "agent_values"

        for peer in peers:
            peer.n_peers = len(peers)
            peer.group = self

            # setup agent values
            if use_agent_values:
                peer.peer_values[key] = self.agent_values  # noqa (Eq. 6)
                peer.peer_value_functions[key] = self._update_agent_values

    def _update_agent_values(self, batch_size=10):
        """ Updates the agent values with samples from the peers' buffers"""
        targets = np.zeros_like(self.peers, dtype=np.float32)
        counts = np.zeros_like(self.peers, dtype=np.float32)

        for peer in self.peers:
            bs = batch_size // len(self.peers)
            # reward, action, peer, new_obs, old_obs
            if peer.buffer is not None:
                batch = peer.buffer.sample(bs)
                if batch is None:  # buffer not sufficiently full
                    return

                obs = np.array([b[3] for b in batch]).reshape(bs, -1)
                v = peer.value(obs)

                if self.use_advantage:
                    # previous observations
                    prev_obs = np.array([b[4] for b in batch]).reshape(bs, -1)
                    prev_v = peer.value(prev_obs)
                else:
                    prev_v = np.zeros_like(v)  # no advantage (see Eq. 5)

                for i in range(len(batch)):  # Eq. 8
                    target = (batch[i][0] + peer.gamma * v[i]) - prev_v[i]
                    counts[batch[i][2]] += 1
                    targets[batch[i][2]] += target

        # ensure counts are >= 1, don't change these values
        targets[counts == 0] = self.agent_values[counts == 0]
        counts[counts == 0] = 1

        targets /= counts
        self.agent_values += self.lr * (targets - self.agent_values)  # Eq. 7

    def learn(self, n_epochs, max_epoch_len, callbacks, **kwargs):
        """ The outer peer learning routine. """
        assert len(callbacks) == len(self.peers)
        # more solo epochs
        boost_single = 0 < self.switch_ratio < 1
        if boost_single:
            self.switch_ratio = 1 / self.switch_ratio

        self.solo_epoch = False
        peer_epochs = 0
        for i in range(n_epochs):
            # don't do peer learning forever
            if peer_epochs < self.max_peer_epochs:
                # ratio of 0 never performs a solo episode
                if (i % (1 + self.switch_ratio) == 1) ^ boost_single:
                    self.solo_epoch = True
                else:
                    peer_epochs += 1
            else:  # budget spent
                self.solo_epoch = True

            for p, peer, callback in zip(it.count(), self.peers, callbacks):
                self.active_peer = p
                peer.learn(self.solo_epoch, total_timesteps=max_epoch_len,
                           callback=callback, tb_log_name=f"Peer{p}",
                           reset_num_timesteps=False,
                           log_interval=None, **kwargs)
                # update epoch for temperature decay
                peer.epoch += 1

        self.active_peer = None

    def __len__(self):
        return len(self.peers)


def make_peer_class(cls: Type[OffPolicyAlgorithm]):
    """ Creates a mixin with the corresponding algorithm class.
    :param cls: The learning algorithm (needs to have a callable critic).
    :return: The mixed in peer agent class.
    """

    class Peer(cls, ABC):
        """ Abstract Peer class
        needs to be mixed with a suitable algorithm. """
        def __init__(self, temperature, temp_decay, algo_args, env,
                     use_trust=False, use_critic=False, init_trust_values=200,
                     buffer_size=1000, follow_steps=10, seed=None,
                     use_trust_buffer=True, solo_training=False,
                     peers_sample_with_noise=False,
                     sample_random_actions=False, sample_from_suggestions=True,
                     epsilon=0.0, env_args=None, only_follow_peers=False):
            if env_args is None:
                env_args = {}
            super(Peer, self).__init__(**algo_args,
                                       env=make_env(env, **env_args),
                                       seed=seed)
            # create noise matrix on the correct device
            if hasattr(self.actor, "reset_noise"):
                self.actor.reset_noise(self.env.num_envs)

            self.solo_training = solo_training
            self.init_values = dict()
            # store all peer values, e.g., trust and agent values in a dict
            self.peer_values = dict()
            # store corresponding functions as well
            self.peer_value_functions = dict()

            self.buffer = SuggestionBuffer(buffer_size)
            self.followed_peer = None
            self.__n_peers = None
            self.group = None
            self.epoch = 0

            if sample_random_actions:
                epsilon = 1.0

            if not solo_training:
                # all peers suggest without noise
                self.peers_sample_with_noise = peers_sample_with_noise
                # actions are sampled instead of taken greedily
                self.sample_actions = sample_from_suggestions
                self.epsilon = epsilon
                self.use_critic = use_critic

                if use_trust:
                    self.trust_values = np.array([])
                    self.init_values["trust"] = init_trust_values
                    self.peer_value_functions["trust"] = self._update_trust

                    self.use_buffer_for_trust = use_trust_buffer

                # sampling parameters
                self.temperature = temperature
                self.temp_decay = temp_decay

                self.follow_steps = follow_steps
                self.steps_followed = 0

                self.only_follow_peers = only_follow_peers

        @property
        def n_peers(self):
            return self.__n_peers

        @n_peers.setter
        def n_peers(self, n_peers):
            self.__n_peers = n_peers

            # Also reset the trust values
            if "trust" in self.init_values.keys():
                self.trust_values = np.full(self.__n_peers,
                                            self.init_values["trust"],
                                            dtype=np.float32)
                self.peer_values["trust"] = self.trust_values

        def critique(self, observations, actions) -> np.array:
            """ Evaluates the actions with the critic. """
            with torch.no_grad():
                a = torch.as_tensor(actions, device=self.device)
                o = torch.as_tensor(observations, device=self.device)

                # Compute the next Q values: min over all critic targets
                q_values = torch.cat(self.critic(o, a), dim=1)  # noqa
                q_values, _ = torch.min(q_values, dim=1, keepdim=True)
                return q_values.cpu().numpy()

        def get_action(self, obs, deterministic=False):
            """ The core function of peer learning acquires the suggested
            actions of the peers and chooses one based on the settings. """
            # follow peer for defined number of steps
            followed_steps = self.steps_followed
            self.steps_followed += 1
            self.steps_followed %= self.follow_steps
            if 0 < followed_steps:
                peer = self.group.peers[self.followed_peer]
                det = (peer != self and not self.peers_sample_with_noise) or \
                    deterministic
                action, _ = peer.policy.predict(obs, deterministic=det)
                return action, None

            # get actions
            actions = []
            for peer in self.group.peers:
                # self always uses exploration, the suggestions of the other
                # peers only do if the critic method isn't used.
                det = (peer != self and not self.peers_sample_with_noise) or \
                      deterministic
                action, _ = peer.policy.predict(obs, deterministic=det)
                actions.append(action)
            actions = np.asarray(actions).squeeze(1)

            # critic (Eq. 3)
            if self.use_critic:
                observations = np.tile(obs, (self.n_peers, 1))
                q_values = self.critique(observations, actions).reshape(-1)
                self.peer_values['critic'] = q_values  # part of Eq. 9

            # calculate peer values, e.g., trust and agent values
            values = np.zeros(self.n_peers)
            for key in self.peer_values.keys():
                # part of Eq. 9 incl. Footnote 7
                values += self.__normalize(self.peer_values[key])

            if self.sample_actions:
                # sample action from probability distribution (Eq. 2)
                temp = self.temperature * np.exp(-self.temp_decay * self.epoch)
                p = np.exp(values / temp)
                p /= np.sum(p)
                self.followed_peer = np.random.choice(self.n_peers, p=p)
            elif self.only_follow_peers:
                p = np.full(self.n_peers, 1 / (self.n_peers - 1))
                p[self.group.peers.index(self)] = 0
                self.followed_peer = np.random.choice(self.n_peers, p=p)
            else:
                # act (epsilon) greedily
                if np.random.random(1) >= self.epsilon:
                    self.followed_peer = np.argmax(values)
                else:
                    self.followed_peer = np.random.choice(self.n_peers)

            action = actions[self.followed_peer].reshape(1, -1)

            return action, None

        @staticmethod
        def __normalize(values):
            """ Normalize the values based on their absolute maximum. """
            return values / np.max(np.abs(values))

        def value(self, observations) -> np.ndarray:
            """ Calculates the value of the observations. """
            actions, _ = self.policy.predict(observations, False)
            return self.critique(observations, actions)

        def _update_trust(self, batch_size=10):
            """ Updates the trust values with samples from the buffer.
                (Eq. 5 and 8)
            """
            if self.use_buffer_for_trust:
                batch = self.buffer.sample(batch_size)
            else:
                batch = self.buffer.latest()
                batch_size = 1
            if batch is None:  # buffer not sufficiently full
                return

            # next observations
            obs = np.array([b[3] for b in batch]).reshape(batch_size, -1)
            v = self.value(obs)

            if self.group.use_advantage:
                # previous observations
                prev_obs = np.array([b[4] for b in batch]).reshape(batch_size,
                                                                   -1)
                prev_v = self.value(prev_obs)
            else:
                prev_v = np.zeros_like(v)  # no comparison to own act (Eq. 5)

            targets = np.zeros(self.n_peers)
            counts = np.zeros(self.n_peers)
            for i in range(batch_size):
                target = (batch[i][0] + self.gamma * v[i]) - prev_v[i]  # Eq. 8
                counts[batch[i][2]] += 1
                targets[batch[i][2]] += target

            # ensure counts are >= 1, don't change these values
            targets[counts == 0] = self.trust_values[counts == 0]
            counts[counts == 0] = 1

            targets /= counts
            # Eq. 4
            self.trust_values += self.group.lr * (targets - self.trust_values)

        def _on_step(self):
            """ Adds updates of the peer values, e.g., trust or agent
            values. """
            super(Peer, self)._on_step()  # noqa

            if not self.group.solo_epoch:
                # update values, e.g., trust and agent values after ever step
                for key in self.peer_value_functions.keys():
                    self.peer_value_functions[key]()

        def _store_transition(self, replay_buffer, buffer_action, new_obs,
                              reward, dones, infos):
            """ Adds suggestion buffer handling. """

            # get previous observations
            old_obs = self._last_obs

            super(Peer, self)._store_transition(replay_buffer,  # noqa
                                                buffer_action, new_obs,
                                                reward, dones, infos)

            if not self.group.solo_epoch:
                # store transition in suggestion buffer as well
                self.buffer.add(reward, buffer_action, self.followed_peer,
                                new_obs, old_obs)

        def _predict_train(self, observation, state=None,
                           episode_start=None, deterministic=False):
            """ The action selection during training involves the peers. """
            if deterministic:
                return self.policy.predict(observation, state=state,
                                           episode_start=episode_start,
                                           deterministic=deterministic)
            else:
                return self.get_action(observation)

        def learn(self, solo_episode=False, **kwargs):
            """ Adds action selection with help of peers. """
            predict = self.predict  # safe for later

            # use peer suggestions only when wanted
            if not (self.solo_training or solo_episode):
                self.predict = self._predict_train
            else:
                self.followed_peer = self.group.peers.index(self)

            result = super(Peer, self).learn(**kwargs)

            self.predict = predict  # noqa
            return result

        def _excluded_save_params(self):
            """ Excludes attributes that are functions. Otherwise, the save
            method fails. """
            ex_list = super(Peer, self)._excluded_save_params()
            ex_list.extend(["peer_value_functions", "peer_values",
                            "group", "predict"])
            return ex_list

    return Peer
