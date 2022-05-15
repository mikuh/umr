import tensorflow as tf
from tensorboardX import SummaryWriter
import multiprocessing as mp
from threading import Thread
from collections import defaultdict
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

    class PredictBatch(object):
        def __init__(self, max_len):
            self.clients = []
            self.states = []
            self.size = 0
            self.max_size = max_len

        def add(self, client, state):
            self.clients.append(client)
            self.states.append(state)
            self.size += 1

        def reset(self):
            self.clients = []
            self.states = []
            self.size = 0

        def full(self):
            return self.size == self.max_size

        @tf.function
        def predict(self, model):
            return model(np.array(self.states))

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
        self.send_queue = queue.Queue(maxsize=args.batch_size * 8 * 2)
        self.args = args

        def send_loop():
            while True:
                msg = self.send_queue.get()
                self.s2c_socket.send_multipart(msg, copy=False)

        self.send_thread = Thread(target=send_loop)
        self.send_thread.setDaemon(True)
        self.send_thread.start()

        self.queue = queue.Queue(maxsize=args.epoch_size * 5)
        self.predict_batch = self.PredictBatch(args.predict_batch_size)
        self.score = deque(maxlen=50)

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
            self.predict_batch.add(client, state)
            if self.predict_batch.full():
                # predict the action, put in the send_queue
                # collect the experience generate the train batch for update model
                distrib_batch, value_batch = self.predict_batch.predict(model=self.model)
                for distrib, value, client in zip(distrib_batch, value_batch, self.predict_batch.clients):
                    action = np.random.choice(self.args.action_size, p=distrib.numpy())
                    self.send_queue.put([client.ident, pickle.dumps(action)])
                    if len(client.memory) > 0:
                        client.memory[-1].reward = reward
                        if done:
                            # should clear client's memory and put to queue
                            self._parse_memory(0, client, True)
                        else:
                            if len(client.memory) == self.args.local_time_max + 1:
                                R = client.memory[-1].value
                                self._parse_memory(R, client, False)
                    client.memory.append(
                        Experience(state=state, action=action, reward=None, value=value[0],
                                   action_prob=distrib[action]))
                self.predict_batch.reset()

    def _parse_memory(self, init_r, client, done):
        mem = client.memory
        if not done:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for k in mem:
            client.score += k.reward
            R = np.clip(k.reward, -1, 1) + self.args.gamma * R
            # advantage = R - k.value
            self.queue.put([k.state, k.action, R, k.action_prob])

        if not done:
            client.memory = [last]
        else:
            client.memory = []
            self.score.append(client.score)
            client.score = 0

    def get_training_dataflow(self):
        def gen():
            while True:
                state, action, target_value, action_prob = self.queue.get()
                yield state, action, target_value, action_prob

        return tf.data.Dataset.from_generator(gen,
                                              output_signature=(tf.TensorSpec(shape=(84, 84, 3, 4), dtype=tf.int32),
                                                                tf.TensorSpec(shape=(), dtype=tf.int32),
                                                                tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                )).batch(
            self.args.batch_size)  # .cache().prefetch(tf.data.AUTOTUNE)

    @tf.function()
    def __train_step(self, data):
        state, action, target_value, action_prob = data
        with tf.GradientTape() as tape:
            policy, value = self.model(state)
            adv = tf.subtract(target_value, tf.stop_gradient(value))
            log_probs = tf.math.log(policy + 1e-6)
            log_pi_a_given_s = tf.reduce_sum(log_probs * tf.one_hot(action, self.args.action_size), 1)
            pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, self.args.action_size), 1)
            importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))
            policy_loss = tf.reduce_sum(log_pi_a_given_s * adv * importance)
            entropy_loss = tf.reduce_sum(policy * log_probs)
            value_loss = tf.nn.l2_loss(tf.subtract(target_value, value))
            loss = tf.reduce_mean([-policy_loss, entropy_loss * 0.01, value_loss])
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, policy_loss, value_loss, tf.reduce_mean(tf.square(adv))

    def learn(self):
        dataset = self.get_training_dataflow()
        step = 0
        for epoch in range(1, 100):
            step += 1
            for data in tqdm(dataset.take(self.args.epoch_size), total=self.args.epoch_size, desc=f"EPOCH {epoch}"):
                loss, policy_loss, value_loss, advantage = self.__train_step(data)
                self.writer.add_scalar('train/loss', loss.numpy(), step)
                self.writer.add_scalar('train/policy_loss', policy_loss.numpy(), step)
                self.writer.add_scalar('train/value_loss', value_loss.numpy(), step)
                self.writer.add_scalar('train/advantage', advantage.numpy(), step)
            mean_score = sum(self.score) / 50
            max_score = max(self.score)
            logger.info(f"EPOCH:{epoch}, Mean Score: {mean_score}, Max Score: {max_score}")
            self.writer.add_scalar('score/mean_score', mean_score, step)
            self.writer.add_scalar('score/max_score', max_score, step)
            self.model.save_weights(os.path.join(self.log_dir, 'checkpoints'))


if __name__ == '__main__':
    from umr.a3c.model import A3C

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', help='env name', default='ALE/Breakout-v5')
    parser.add_argument('--workers', default=mp.cpu_count()*2)
    parser.add_argument('--frame_history', '--history', default=4)
    parser.add_argument('--render_mode', '--em', help='env mode', default=None)
    parser.add_argument('--url_c2s', help='zmq pipeline url c2s', default='ipc://agent-c2s')
    parser.add_argument('--url_s2c', help='zmq pipeline url s2c', default='ipc://agent-s2c')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--predict_batch_size', default=16)
    parser.add_argument('--epoch_size', default=6000)
    parser.add_argument('--local_time_max', default=5)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--log_dir', default='train_log')
    args = parser.parse_args()
    args.action_size = get_gym_env(args.env_name).action_space.n
    logger.info(args)

    workers = [AgentWorker(i, args) for i in range(args.workers)]
    [w.start() for w in workers]
    master = AgentMaster(args)
    master.start()
    master.learn()
