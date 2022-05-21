import tensorflow as tf
from tensorboardX import SummaryWriter
import multiprocessing as mp
from threading import Thread
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pickle
import zmq
import queue
import gym
import numpy as np
import argparse
from tqdm import tqdm
from collections import deque
import os
from umr.utils import logger
from umr.utils import get_gym_env


class Experience(object):

    def __init__(self, state, action, reward, **kwargs):
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in kwargs.items():
            setattr(self, k, v)


class AgentWorker(mp.Process):

    def __init__(self, index: int, args):
        super(AgentWorker, self).__init__()
        self.id = int(index)
        self.name = f'worker-{self.id}'
        self.identity = self.name.encode('utf-8')
        self.args = args
        print(self.identity)

    def run(self):
        # zmq socket
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.args.url_c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        s2c_socket.connect(self.args.url_s2c)

        env = self.__get_env(self.args)
        state = env.reset()
        reward, done = 0, False
        while True:
            c2s_socket.send(pickle.dumps((self.identity, state, reward, done)), copy=False)
            action = pickle.loads(s2c_socket.recv(copy=False))
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()

    def __get_env(self, args) -> gym.Env:
        return get_gym_env(args.env_name, render_mode=args.render_mode)


class AgentMaster(Thread):
    class ClientState(object):
        def __init__(self):
            self.memory = []  # list of Experience
            self.ident = None
            self.score = 0
            self.steps = 0

    class Predictor(object):

        def __init__(self, workers, model):
            self.pool = ThreadPoolExecutor(max_workers=workers)
            self.model = model

        @tf.function
        def _predict(self, state):
            # self.model.make_predict_function()
            logits, value = self.model(state)
            distrib = tf.nn.softmax(logits)
            action = tf.random.categorical(logits, 1)[0, 0]
            return distrib[0], value[0], action, distrib[0, action]

        def put_task(self, state, callback):
            output = self.pool.submit(self._predict, state)
            output.add_done_callback(callback)

    def __init__(self, args):
        super(AgentMaster, self).__init__()
        self.model = A3C(args.action_size)
        self.args = args
        self.daemon = True
        self.name = 'Master'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        # zmq socket
        self.context = zmq.Context()
        # receive the state, reward, done
        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(args.url_c2s)
        self.c2s_socket.set_hwm(10)
        # send the action
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(args.url_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=args.batch_size)
        self.args = args

        def send_loop():
            while True:
                msg = self.send_queue.get()
                self.s2c_socket.send_multipart(msg, copy=False)

        self.send_thread = Thread(target=send_loop)
        self.send_thread.setDaemon(True)
        self.send_thread.start()

        self.queue = queue.Queue(maxsize=args.batch_size * args.predict_thread)
        self.score = deque(maxlen=50)
        self.episode_steps = deque(maxlen=50)
        self.episode = 0
        self.predictors = self.Predictor(args.predict_thread, self.model)

        self.log_dir = os.path.join(args.log_dir, f"train-{args.env_name}")
        self.writer = SummaryWriter(self.log_dir)

    def run(self) -> None:
        self.clients = defaultdict(self.ClientState)
        while True:
            msg = pickle.loads(self.c2s_socket.recv(copy=False))
            ident, state, reward, done = msg
            client = self.clients[ident]
            if client.ident is None:
                client.ident = ident
            # process message
            if len(client.memory) > 0:
                client.memory[-1].reward = reward
                if done:
                    # should clear client's memory and put to queue
                    self._parse_memory(0, client, True)
                else:
                    if len(client.memory) == self.args.local_time_max + 1:
                        R = client.memory[-1].value
                        self._parse_memory(R, client, False)

            self._collect_experience(state, client)

    def _collect_experience(self, state, client):
        def cb(output):
            distrib, value, action, action_prob = output.result()
            client.memory.append(
                Experience(state=state, action=action, reward=None, value=value, action_prob=action_prob))
            self.send_queue.put([client.ident, pickle.dumps(action.numpy())])

        self.predictors.put_task(state[np.newaxis, :], cb)

    def _parse_memory(self, init_r, client, done):
        mem = client.memory
        if not done:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for k in mem:
            client.score += k.reward
            client.steps += 1
            R = np.clip(k.reward, -1, 1) + self.args.gamma * R
            # advantage = R - k.value
            self.queue.put([k.state, k.action, R, k.action_prob])

        if not done:
            client.memory = [last]
        else:
            self.score.append(client.score)
            self.episode_steps.append(client.steps)
            self.episode += 1
            client.memory = []
            client.score = 0
            client.steps = 0

    def get_training_dataflow(self):
        def gen():
            while True:
                state, action, target_value, action_prob = self.queue.get()
                yield state, action, target_value, action_prob

        return tf.data.Dataset.from_generator(gen,
                                              output_signature=(tf.TensorSpec(shape=(84, 84, 3, 4), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(), dtype=tf.int32),
                                                                tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                )).batch(
            self.args.batch_size).prefetch(self.args.epoch_size)  # .cache().prefetch(tf.data.AUTOTUNE)

    @tf.function
    def __train_step(self, data, epoch):
        state, action, target_value, action_prob = data
        with tf.GradientTape() as tape:
            logits, value = self.model(state)
            policy = tf.nn.softmax(logits)
            value = tf.squeeze(value, [1])  # (B,)
            advantage = tf.subtract(target_value, tf.stop_gradient(value))
            log_probs = tf.math.log(policy + 1e-6)
            log_pi_a_given_s = tf.reduce_sum(log_probs * tf.one_hot(action, self.args.action_size), 1)  # (B,)
            pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, self.args.action_size), 1)
            importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))
            policy_loss = -tf.reduce_mean(log_pi_a_given_s * advantage * importance)
            entropy_loss = tf.reduce_mean(policy * log_probs) * get_beta(epoch)
            value_loss = tf.nn.l2_loss(value - target_value) / self.args.batch_size
            loss = tf.add_n([policy_loss, entropy_loss, value_loss])
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, policy_loss, value_loss, tf.reduce_mean(advantage), tf.reduce_mean(importance), \
               entropy_loss, tf.reduce_mean(value)

    def learn(self):
        dataset = self.get_training_dataflow()
        step = 0
        best_score = 0
        for epoch in range(1, 600):
            for data in tqdm(dataset.take(self.args.epoch_size), total=self.args.epoch_size, desc=f"Epoch {epoch}"):
                step += 1
                loss, policy_loss, value_loss, advantage, importance, entropy_loss, value = self.__train_step(data,
                                                                                                              epoch)
            mean_score = np.mean(self.score)
            max_score = max(self.score)
            logger.info(f"EPOCH:{epoch}, Mean Score: {mean_score}, Max Score: {max_score}")
            self.writer.add_scalar('train/loss', loss.numpy(), step)
            self.writer.add_scalar('train/policy_loss', policy_loss.numpy(), step)
            self.writer.add_scalar('train/value_loss', value_loss.numpy(), step)
            self.writer.add_scalar('train/advantage', advantage.numpy(), step)
            self.writer.add_scalar('train/importance', importance.numpy(), step)
            self.writer.add_scalar('train/entropy_loss', entropy_loss.numpy(), step)
            self.writer.add_scalar('client/mean_score', mean_score, step)
            self.writer.add_scalar('client/max_score', max_score, step)
            self.writer.add_scalar('client/episode_steps', np.mean(self.episode_steps), self.episode)
            self.writer.add_scalar('client/queue_length', self.queue.qsize(), step)
            if mean_score > best_score:
                self.model.save_weights(os.path.join(self.log_dir, 'checkpoints'))
                best_score = mean_score



def get_beta(epoch):
    if epoch < 30:
        return 0.002
    elif epoch < 60:
        return 0.001
    elif epoch < 90:
        return 0.0005
    elif epoch < 120:
        return 0.0002
    else:
        return 0.0001


if __name__ == '__main__':
    from umr.a3c.model import A3C

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', help='env name', default='ALE/Breakout-v5')
    parser.add_argument('--workers', default=mp.cpu_count())
    parser.add_argument('--frame_history', '--history', default=4)
    parser.add_argument('--render_mode', '--em', help='env mode', default=None)
    parser.add_argument('--url_c2s', help='zmq pipeline url c2s', default='ipc://agent-c2s')
    parser.add_argument('--url_s2c', help='zmq pipeline url s2c', default='ipc://agent-s2c')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--predict_thread', default=4)
    parser.add_argument('--epoch_size', default=1000)
    parser.add_argument('--local_time_max', default=5)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lr', default=tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000,
                                                                                       decay_rate=0.95, staircase=True))
    parser.add_argument('--log_dir', default='train_log')
    args = parser.parse_args()
    args.action_size = get_gym_env(args.env_name).action_space.n
    logger.info(args)

    workers = [AgentWorker(i, args) for i in range(args.workers)]
    [w.start() for w in workers]
    master = AgentMaster(args)
    master.start()
    master.learn()
