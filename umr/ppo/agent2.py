import tensorflow as tf
import ray
import numpy as np
from collections import deque
from umr.ppo.model import create_model
from umr.utils import get_gym_env

ray.init()


class RolloutInstance(object):

    def __init__(self, max_size, gamma, lamda, gae_normal=True):
        self.max_size = max_size
        self.gae_normal = gae_normal
        self.gamma = gamma
        self.lamda = lamda
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob)
        self.values.append(value)
        self.dones.append(done)

    def reset(self):
        self.states = [self.states[-1]]
        self.actions = [self.actions[-1]]
        self.rewards = [self.rewards[-1]]
        self.probs = [self.probs[-1]]
        self.values = [self.values[-1]]
        self.dones = [self.dones[-1]]

    def parse_instance(self):
        states = self.states[:-1]
        actions = self.actions[:-1]
        values = self.values[:-1]
        next_values = self.values[1:]
        rewards = self.rewards[:-1]
        dones = self.dones[:-1]
        probs = self.probs[:-1]
        advs = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        advs = np.stack(advs)
        gaes = copy.deepcopy(advs)
        for t in reversed(range(len(advs) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lamda * gaes[t + 1]
        targets = gaes + values
        if self.gae_normal:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        self.reset()
        return dict(state=states, action=actions, prob=probs, gae=gaes.tolist(), target=targets.tolist(),
                    done=dones)


@ray.remote
class ClientRecord(object):

    def __init__(self):
        self.scores = deque(maxlen=50)
        self.step_per_episode = deque(maxlen=50)
        self.episodes = 0

    def add(self, score):
        self.scores.append(score)

    def mean_score(self):
        return np.mean(self.scores)

    def max_score(self):
        return max(self.scores)


@ray.remote
class RolloutWorker(object):

    def __init__(self):
        self.env = get_gym_env("ALE/Breakout-v5")
        self.model = create_model(self.env.action_space.n)
        self.state = self.env.reset()
        self.rollout = 5
        self.score = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def update_weights(self, w):
        self.model.set_weights(w)

    def run_env(self):
        for _ in range(self.rollout):
            action, distribute, value = self.predict(self.state[np.newaxis, :])
            next_state, reward, done, _ = self.step(action.numpy())
            self.score += reward
            if done:
                next_state = self.reset()
                self.score = 0
            self.state = next_state

    @tf.function
    def predict(self, state):
        logits, value = self.model(state)
        distrib = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0]
        return action, distrib[0], value[0, 0]


workers = [RolloutWorker.remote() for _ in range(4)]

# print(ray.get(results))
