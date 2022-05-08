import gym
import cv2
from umr.atari_wrapper import FireResetEnv, MapState, FrameStack, LimitLength


def get_gym_env(env_name: str, is_train: bool = True, image_size: tuple = (84, 84)) -> gym.Env:
    env = gym.make(env_name)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, image_size))
    env = FrameStack(env, 4)
    if is_train:
        env = LimitLength(env, 60000)
    return env
