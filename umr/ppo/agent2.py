import tensorflow as tf
import ray
import copy
import numpy as np
import argparse
from collections import deque
from umr.ppo.model import create_model
from umr.utils import get_gym_env
from umr.utils import logger

ray.init()


class Experience(object):

    def __init__(self, state, action, reward, **kwargs):
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in kwargs.items():
            setattr(self, k, v)


class ClientMemory(object):

    def __init__(self, args):
        self.gae_normal = args.gae_normal
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.experiences = []

    def reset(self):
        self.experiences = self.experiences[-1]

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

    def parse_memory(self):
        R = self.experiences[-1].value
        experiences = self.experiences
        if not self.experiences[-1].done:
            last = self.experiences[-1]
            experiences = self.experiences[:-1]
        else:
            R = 0
        experiences.reverse()
        for e in experiences:
            R = np.clip(e.reward, -1, 1) + self.args.gamma * R
            adv = R - e.value


@ray.remote
class ReplyBuffer(object):

    def __int__(self):
        self.experiences = []

    def store(self, state, action, pi_old, gae, target):
        self.experiences.append((state, action, pi_old, gae, target))

    def sampling(self):
        sample_range = np.arange(len(self.action))
        np.random.shuffle(sample_range)
        sample_idx = sample_range[:self.args.batch_size]

    def clear(self):
        pass


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

    def __init__(self, args):
        self.env = get_gym_env(args.env_name)  # "ALE/Breakout-v5"
        self.model = create_model(self.env.action_space.n)
        self.memory = ClientMemory(args)
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
        while len(self.memory.experiences) < self.rollout + 1:
            action, distribute, value = self.predict(self.state[np.newaxis, :])
            next_state, reward, done, _ = self.step(action.numpy())
            self.score += reward
            if done:
                self.state = self.reset()
                self.score = 0
                return
            self.memory.experiences.append(
                Experience(state=self.state, action=action, reward=reward, action_prob=distribute, value=value))
            self.state = next_state
        return

    @tf.function
    def predict(self, state):
        logits, value = self.model(state)
        distrib = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0]
        return action, distrib[0], value[0, 0]


# workers = [RolloutWorker.remote() for _ in range(4)]
#
# print(ray.get(results))
# def gen():
#     for a, b in [[[1, 2, 3], 2], [[3, 4, 5], 4], [[5, 6, 7], 6]]:
#         yield a, b

#
# dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(3,), dtype=tf.int32),
#                                                                 tf.TensorSpec(shape=(), dtype=tf.int32)))
# print(list(dataset.take(1)))


dataset = tf.data.Dataset.from_slices([[[1, 2, 3], 2], [[3, 4, 5], 4], [[5, 6, 7], 6]])
print(list(dataset.as_numpy_iterator()))