import numpy as np

from manual_peer import ManualPeer


class RoomExpert(ManualPeer):
    def _predict(self, obs, **kwargs):
        pos = obs[:len(obs) // 2]
        goal = obs[len(obs) // 2:]

        if np.all(pos == goal):
            dim = np.argmax(np.abs(goal - 0.5))
            return 2 * dim + 1 - int(goal[dim])
        else:
            actions = []
            for i in range(len(obs) // 2):
                if pos[i] < goal[i]:
                    actions.append(2 * i)
                elif pos[i] > goal[i]:
                    actions.append(2 * i + 1)

            return self.np_random.choice(actions)


class RoomAdversarial(RoomExpert):
    def _predict(self, obs, **kwargs):
        state = np.copy(obs)
        state[len(obs) // 2:] = 1 - np.round(state[len(obs) // 2:])
        return super()._predict(state)


if __name__ == '__main__':
    import gym
    import env as blabla  # noqa
    env = gym.make("Room-v21", length=21, dims=3)
    # model = RoomExpert()
    model = RoomAdversarial()

    rewards = []

    for _ in range(1):
        obs_ = env.reset()  # noqa

        done = False
        reward = 0
        old = 0  # noqa

        while not done:
            pos_ = obs_[:len(obs_) // 2]
            goal_ = obs_[len(obs_) // 2:]

            new = np.sum(np.abs(pos_ - goal_))

            print(new, new - old >= 0, env.agent_location,
                  env.goal_location)
            old = new

            a = model._predict(obs_)  # noqa
            obs_, r, done, info = env.step(a)
            reward += r
        rewards.append(reward)

    print(np.mean(rewards))
