import tensorflow as tf
import ray
import copy
import os
import numpy as np
import zmq
import pickle
import multiprocessing as mp
import argparse
from threading import Thread
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import deque
from umr.ppo.model import create_model
from umr.utils import get_gym_env
from umr.utils import logger


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

    def add_experience(self, exp):
        self.experiences.append(exp)

    def parse_memory(self, replay_buffer):
        experiences = self.experiences
        R = experiences[-1].value
        done = experiences[-1].done
        if not done:
            last = self.experiences[-1]
            experiences = self.experiences[:-1]
        else:
            R = 0
        experiences.reverse()
        targets = []
        advs = []
        for e in experiences:
            R = np.clip(e.reward, -1, 1) + self.gamma * R
            adv = R - e.value
            targets.append(R)
            advs.append(adv)
        gaes = copy.deepcopy(advs)
        for i in range(1, len(gaes)):
            gaes[i] = gaes[i] + self.gamma * self.lamda * gaes[i - 1]
        for e, gae, target in zip(experiences, gaes, targets):
            replay_buffer.store.remote(e.state, e.action, e.action_prob, gae, target)

        if not done:
            self.experiences = [last]
        else:
            self.experiences = []


@ray.remote
class ReplyBuffer(object):

    def __init__(self):
        self.scores = deque(maxlen=50)
        self.step_per_episode = deque(maxlen=50)

        self.state = []
        self.action = []
        self.pi_old = []
        self.gae = []
        self.target = []

    def store(self, state, action, pi_old, gae, target):
        self.state.append(state)
        self.action.append(action)
        self.pi_old.append(pi_old)
        self.gae.append(gae)
        self.target.append(target)

    def sampling(self):
        state = np.asarray(self.state)
        action = np.asarray(self.action)
        pi_old = np.asarray(self.pi_old, dtype=np.float32)
        gae = np.asarray(self.gae, dtype=np.float32)
        target = np.asarray(self.target, dtype=np.float32)
        self.clear()
        return state, action, pi_old, gae, target

    def clear(self):
        self.state = []
        self.action = []
        self.pi_old = []
        self.gae = []
        self.target = []

    def add_client_record(self, score, steps):
        self.scores.append(score)
        self.step_per_episode.append(steps)

    def score(self):
        return np.mean(self.scores), max(self.scores)

    def get_state(self):
        return self.state


@ray.remote
class AgentWorker(object):
    class ClientMemory(object):

        def __init__(self, args):
            self.gae_normal = args.gae_normal
            self.gamma = args.gamma
            self.lamda = args.lamda
            self.experiences = []

        def add_experience(self, exp):
            self.experiences.append(exp)

        def parse_memory(self, replay_buffer):
            experiences = self.experiences
            R = experiences[-1].value
            done = experiences[-1].done
            if not done:
                last = self.experiences[-1]
                experiences = self.experiences[:-1]
            else:
                R = 0
            experiences.reverse()
            targets = []
            advs = []
            for e in experiences:
                R = np.clip(e.reward, -1, 1) + self.gamma * R
                adv = R - e.value
                targets.append(R)
                advs.append(adv)
            gaes = copy.deepcopy(advs)
            for i in range(1, len(gaes)):
                gaes[i] = gaes[i] + self.gamma * self.lamda * gaes[i - 1]
            for e, gae, target in zip(experiences, gaes, targets):
                replay_buffer.store.remote(e.state, e.action, e.action_prob, gae, target)

            if not done:
                self.experiences = [last]
            else:
                self.experiences = []

    def __init__(self, args):
        self.env = get_gym_env(args.env_name)  # "ALE/Breakout-v5"
        self.memory = self.ClientMemory(args)
        self.state = self.env.reset()
        self.rollout = args.rollout
        self.score = 0
        self.steps = 0

        # zmq socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(args.url)

    def reset(self):
        self.score = 0
        self.steps = 0
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.score += reward
        self.steps += 1
        return next_state, reward, done, info

    def update_weights(self, w):
        self.model.set_weights(w)

    def run_env(self, replay_buffer):
        while len(self.memory.experiences) < self.rollout + 1:
            self.socket.send(pickle.dumps(self.state[np.newaxis, :]))
            action, distribute, value = pickle.loads(self.socket.recv())
            next_state, reward, done, _ = self.step(action)
            self.memory.add_experience(
                Experience(state=self.state, action=action, reward=reward, action_prob=distribute,
                           value=value, done=done))
            if done:
                replay_buffer.add_client_record.remote(self.score, self.steps)
                self.state = self.reset()
                # self.memory.parse_memory(replay_buffer)
                break
            self.state = next_state
        self.memory.parse_memory(replay_buffer)



class AgentMaster(object):

    def __init__(self, args):
        self.workers = [AgentWorker.remote(args) for _ in range(args.workers)]
        self.replay_buffer = ReplyBuffer.remote()
        self.model = create_model(action_size=args.action_size)
        self.opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        self.log_dir = os.path.join(args.log_dir, f"train-{args.env_name}")
        self.writer = SummaryWriter(self.log_dir)
        self.args = args

        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.args.url)
        self.poller.register(self.socket, zmq.POLLIN)

        def predict_loop():
            while True:
                msg = self.socket.recv_multipart()
                action, distrib, value = self.predict(pickle.loads(msg[2]))
                msg[2] = pickle.dumps([action.numpy(), distrib.numpy(), value.numpy()])
                self.socket.send_multipart(msg)

        self.predict_thread = Thread(target=predict_loop)
        self.predict_thread.daemon = True
        self.predict_thread.start()

    @tf.function
    def train_step(self, state, action, pi_old, adv, target):
        with tf.GradientTape() as tape:
            logits, value = self.model(state)
            pi = tf.nn.softmax(logits)
            value = tf.squeeze(value, [1])
            entropy_loss = tf.reduce_mean(pi * tf.math.log(pi + 1e-8)) * 0.01
            onehot_action = tf.one_hot(action, self.args.action_size)  # (B, action_size)

            action_prob = tf.reduce_sum(pi * onehot_action, axis=1)
            action_prob_old = tf.reduce_sum(pi_old * onehot_action, axis=1)

            ratio = tf.exp(tf.math.log(action_prob + 1e-8) - tf.math.log(action_prob_old + 1e-8))  # (B,)
            clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - self.args.ppo_eps,
                                             clip_value_max=1 + self.args.ppo_eps)
            minimum = tf.minimum(tf.multiply(adv, clipped_ratio), tf.multiply(adv, ratio))
            pi_loss = -tf.reduce_mean(minimum)
            value_loss = tf.reduce_mean(tf.square(target - value))
            loss = pi_loss + value_loss + entropy_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, pi_loss, value_loss, entropy_loss, tf.reduce_mean(adv), tf.reduce_mean(value), tf.reduce_mean(
            clipped_ratio)

    @tf.function
    def predict(self, state):
        logits, value = self.model(state)
        distrib = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0]
        return action, distrib[0], value[0, 0]

    def learn(self):
        step = 0

        for epoch in range(1, 600):
            for _ in tqdm(range(self.args.epoch_size), total=self.args.epoch_size):
                step += 1
                ray.get([w.run_env.remote(self.replay_buffer) for w in self.workers])
                sampling = self.replay_buffer.sampling.remote()
                _state, _action, _pi_old, _adv, _target = ray.get(sampling)
                for _ in range(self.args.n_update):
                    state, action, pi_old, adv, target = self.batch(_state, _action, _pi_old, _adv, _target)
                    loss, pi_loss, value_loss, entropy, adv, value, clipped_ratio = self.train_step(state, action,
                                                                                                    pi_old,
                                                                                                    adv, target)

            self.writer.add_scalar("train/loss", loss.numpy(), step)
            self.writer.add_scalar('train/policy_loss', pi_loss.numpy(), step)
            self.writer.add_scalar('train/value_loss', value_loss.numpy(), step)
            self.writer.add_scalar('train/advantage', adv.numpy(), step)
            self.writer.add_scalar('train/clipped_ratio', clipped_ratio.numpy(), step)
            self.writer.add_scalar('train/entropy_loss', entropy.numpy(), step)
            mean_score, max_score = ray.get(self.replay_buffer.score.remote())
            self.writer.add_scalar("client/mean_score", mean_score, step)
            self.writer.add_scalar("client/max_score", max_score, step)
            logger.info(f"EPOCH:{epoch}, Mean Score: {mean_score}, Max Score: {max_score}")

    def batch(self, state, action, pi_old, adv, target):
        sample_range = np.arange(len(adv))
        np.random.shuffle(sample_range)
        sample_idx = sample_range[:self.args.batch_size]
        return state[sample_idx], action[sample_idx], pi_old[sample_idx], adv[sample_idx], target[sample_idx]


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '--env', help='env name', default='ALE/Breakout-v5')
parser.add_argument('--render_mode', '--em', help='env mode', default=None)
parser.add_argument('--workers', default=mp.cpu_count())
parser.add_argument('--gae_normal', default=True)
parser.add_argument('--gamma', default=0.99)
parser.add_argument('--lamda', default=0.95)
parser.add_argument('--n_update', default=4)
parser.add_argument('--lr', default=tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000,
                                                                                   decay_rate=0.95, staircase=True))
parser.add_argument('--batch_size', default=128)
parser.add_argument('--epoch_size', default=1000)
parser.add_argument('--ppo_eps', default=0.2)
parser.add_argument('--frame_history', '--history', default=4)
parser.add_argument('--url', help='zmq pipeline url', default='ipc://agent-pipline')

parser.add_argument('--log_dir', default='train_log')
args = parser.parse_args()
args.action_size = get_gym_env(args.env_name).action_space.n
args.rollout = args.batch_size // args.workers + 1
ray.init()
agent = AgentMaster(args)
agent.learn()
