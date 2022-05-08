from multiprocessing import Process
from threading import Thread
import json
import zmq
from umr.utils import get_gym_env


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

    def __int__(self, url_c2s: str, url_s2c: str):
        pass
