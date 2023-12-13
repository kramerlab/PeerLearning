from stable_baselines3 import DQN
from peer import make_peer_class


class DQNPeer(make_peer_class(DQN)):
    """
    A DQN version to be used with peer learning. Therefore, it features
    a critic function
    """
    def critic(self, observations, actions):
        q_values = self.q_net(observations).reshape(len(actions), -1, 1)
        tmp = q_values[range(len(actions)), actions, :]
        return tmp, tmp  # SAC critic outputs multiple values, so this need
        # to do the same

    def get_action(self, *args, **kwargs):
        action, _ = super().get_action(*args, **kwargs)
        return action.reshape(-1), _
