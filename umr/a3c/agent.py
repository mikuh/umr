from multiprocessing import Process
from threading import Thread
from collections import defaultdict
import json
import zmq
import queue
import gym
import numpy as np
from umr.utils import logger
from umr.utils import get_gym_env


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in kwargs.items():
            setattr(self, k, v)


class AgentWorker(Process):

    def __int__(self, env_name: str, pid: int, url_c2s: str, url_s2c: str):
        super(AgentWorker, self).__init__()
        self.env = self.__get_env(env_name)
        self.id = int(pid)
        self.name = f'worker-{self.id}'
        self.identity = self.name.encode('utf-8')

        # zmq socket
        context = zmq.Context()
        self.c2s_socket = context.socket(zmq.PUSH)
        self.c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        self.c2s_socket.set_hwm(2)
        self.c2s_socket.connect(url_c2s)

        self.s2c_socket = context.socket(zmq.DEALER)
        self.s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        self.s2c_socket.connect(url_s2c)

    def run(self):
        state = self.env.reset()
        reward, done = 0, False
        while True:
            self.c2s_socket.send(json.dumps((self.identity, state, reward, done)), copy=False)
            action = json.loads(self.s2c_socket.recv(copy=False))
            state, reward, done, _ = self.env.step(action)
            if done:
                state = self.env.reset()

    def __get_env(self, env_name: str) -> gym.Env:
        return get_gym_env(env_name)


class AgentMaster(Thread):
    class ClientState(object):
        def __init__(self):
            self.memory = []  # list of Experience
            self.ident = None

    def __int__(self, model, url_c2s: str, url_s2c: str, batch_size=128):
        super(AgentMaster, self).__int__()
        self.model = model
        self.daemon = True
        self.name = 'Master'
        # zmq socket
        self.context = zmq.Context()
        # receive the state, reward, done
        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(url_c2s)
        self.c2s_socket.set_hwm(10)
        # send the action
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(url_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=batch_size * 8 * 2)

        def send_loop():
            while True:
                msg = self.send_queue.get()
                self.s2c_socket.send_multipart(msg, copy=False)

        self.send_thread = Thread(target=send_loop)
        self.send_thread.setDaemon(True)
        self.send_thread.start()

        # self.predict_queue = queue.Queue()

    def run(self) -> None:
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = json.loads(self.c2s_socket.recv(copy=False))
                ident, state, reward, done = msg
                client = self.clients[ident]
                if client.ident is None:
                    client.ident = ident
                # predict the action, put in the send_queue
                # collect the experience generate the train batch for update model
                distrib, value = self.model([state])
                action = np.random.choice(len(distrib), p=distrib[0])
                self.send_queue.put([client.ident, json.dumps(action)])
                if len(client.memory) > 0:
                    client.memory[-1].reward = reward
                    if done:
                        # should clear client's memory and put to queue
                        self._parse_memory(0, client, True)
                    else:
                        if len(client.memory) == LOCAL_TIME_MAX + 1:
                            R = client.memory[-1].value
                            self._parse_memory(R, client, False)
                client.memory.append(
                        TransitionExperience(state, action, reward=None, value=value[0], prob=distrib[0][action]))

                # self._process_msg(client, state, reward, done)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    def _process(self):
        pass

    def loss(self):
        pass
