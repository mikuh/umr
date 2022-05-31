import tensorflow as tf
import ray
import os
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import multiprocessing as mp
from collections import deque
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import zmq
import pickle
import copy
from umr.utils import get_gym_env
from umr.ppo.model import PPO
from umr.utils import logger


@ray.remote
class Record(object):

    def __init__(self):
        self.scores = deque(maxlen=50)

    def add(self, score):
        self.scores.append(score)

    def score(self):
        return np.mean(self.scores), max(self.scores)


@ray.remote
class Worker(object):
    class Instance(object):

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

    def __init__(self, iden, args):
        self.env = get_gym_env(args.env_name, render_mode=args.render_mode)
        self.rollout = args.rollout
        self.instance = self.Instance(args.rollout, args.gamma, args.lamda, args.gae_normal)
        self.id = int(iden)
        self.name = f'worker-{self.id}'
        self.identity = self.name.encode('utf-8')
        self.args = args
        self.score = 0

        self.state = self.env.reset()

        # zmq socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.args.url)

    def rollout(self, record):
        for _ in range(self.rollout + 1):
            self.socket.send(pickle.dumps(self.state), copy=False)
            msg = self.socket.recv(copy=False)
            action, action_prob, value = pickle.loads(msg)
            next_state, reward, done, _ = self.env.step(action)
            self.score += reward
            self.instance.add(self.state, action, np.clip(reward, -1, 1), action_prob, value, done)
            if done:
                self.state = self.env.reset()
                record.add.remote(self.score)
                self.score = 0
            else:
                self.state = next_state
        return self.instance.parse_instance()


class Agent(object):

    def __init__(self, args):
        self.workers = [Worker.remote(i, args) for i in range(args.workers)]
        self.model = PPO(action_size=args.action_size)
        self.opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        self.log_dir = os.path.join(args.log_dir, f"train-{args.env_name}")
        self.writer = SummaryWriter(self.log_dir)
        self.args = args
        self.episodes = 0
        self.record = Record.remote()
        # zmq
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(args.url)
        self.poller.register(self.socket, zmq.POLLIN)

        def predict_loop():
            while True:
                states = []
                msgs = []
                for _ in range(args.workers):
                    msg = self.socket.recv_multipart(copy=False)
                    states.append(pickle.loads(msg[2]))
                    msgs.append(msg)
                distrib, value = self.batch_predict(np.asarray(states))
                for msg, d, v in zip(msgs, distrib.numpy(), value.numpy()):
                    a = np.random.choice(self.args.action_size, p=d)
                    msg[2] = pickle.dumps([a, d, v])
                    self.socket.send_multipart(msg, copy=False)

        self.predit_thread = Thread(target=predict_loop)
        self.predit_thread.daemon = True
        self.predit_thread.start()

    @tf.function
    def train_step(self, state, action, pi_old, adv, target):
        with tf.GradientTape() as tape:
            logits, value = self.model(state)
            pi = tf.nn.softmax(logits)
            value = tf.squeeze(value, [1])
            # value = tf.clip_by_value(value, clip_value_min=0, clip_value_max=10)
            entropy_loss = tf.reduce_mean(pi * tf.math.log(pi + 1e-8)) * self.args.entropy_coeff
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

    @tf.function
    def batch_predict(self, state):
        logits, value = self.model(state)
        distrib = tf.nn.softmax(logits)
        return distrib, value

    def learn(self):
        step = 0
        for epoch in range(1, 600):
            for _ in tqdm(range(self.args.epoch_size), total=self.args.epoch_size):
                step += 1
                _state, _action, _pi_old, _adv, _target = self.get_train_data()
                for _ in range(self.args.num_sgd_iter):
                    sample_range = np.arange(len(_action))
                    np.random.shuffle(sample_range)
                    sample_idx = sample_range[:self.args.batch_size]
                    state = _state[sample_idx]
                    action = _action[sample_idx]
                    pi_old = _pi_old[sample_idx]
                    adv = _adv[sample_idx]
                    target = _target[sample_idx]
                    loss, pi_loss, value_loss, entropy, adv, value, clipped_ratio = self.train_step(state, action,
                                                                                                    pi_old,
                                                                                                    adv, target)
            self.writer.add_scalar("train/loss", loss.numpy(), step)
            self.writer.add_scalar('train/policy_loss', pi_loss.numpy(), step)
            self.writer.add_scalar('train/value_loss', value_loss.numpy(), step)
            self.writer.add_scalar('train/advantage', adv.numpy(), step)
            self.writer.add_scalar('train/clipped_ratio', clipped_ratio.numpy(), step)
            self.writer.add_scalar('train/entropy_loss', entropy.numpy(), step)
            mean_score, max_score = ray.get(self.record.score.remote())
            self.writer.add_scalar("client/mean_score", mean_score, step)
            self.writer.add_scalar("client/max_score", max_score, step)
            logger.info(f"EPOCH:{epoch}, Mean Score: {mean_score}, Max Score: {max_score}")

    def get_train_data(self):
        states = []
        actions = []
        probs = []
        gaes = []
        targets = []
        for instance in ray.get([w.rollout.remote(self.record) for w in self.workers]):
            states += instance["state"]
            actions += instance["action"]
            probs += instance["prob"]
            gaes += instance["gae"]
            targets += instance["target"]
            self.episodes += sum(instance["done"])
        return np.array(states), np.array(actions), np.array(probs, dtype=np.float32), np.array(gaes, dtype=np.float32), \
               np.array(targets, dtype=np.float32)


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '--env', help='env name', default='ALE/Breakout-v5')
parser.add_argument('--render_mode', '--em', help='env mode', default=None)
parser.add_argument('--workers', default=mp.cpu_count()*2)
parser.add_argument('--rollout', default=100)
parser.add_argument('--gae_normal', default=True)
parser.add_argument('--gamma', default=0.99)
parser.add_argument('--lamda', default=0.95)
parser.add_argument('--lr', default=tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000,
                                                                                  decay_rate=0.95, staircase=True))

parser.add_argument('--num_sgd_iter', default=10)
parser.add_argument('--entropy_coeff', default=0.005)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--epoch_size', default=1000)
parser.add_argument('--ppo_eps', default=0.2)
parser.add_argument('--frame_history', '--history', default=4)
parser.add_argument('--url', help='zmq pipeline url', default='ipc://agent-pipline')

parser.add_argument('--log_dir', default='train_log')
args = parser.parse_args()
args.action_size = get_gym_env(args.env_name).action_space.n
ray.init()
agent = Agent(args)
agent.learn()
