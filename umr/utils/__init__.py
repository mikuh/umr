import gym
import cv2
from umr.atari_wrapper import FireResetEnv, MapState, FrameStack, LimitLength


def get_gym_env(env_name: str, is_train: bool = True, image_size: tuple = (84, 84), render_mode=None) -> gym.Env:
    if render_mode is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode=render_mode)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, image_size))
    env = FrameStack(env, 4)
    if is_train:
        env = LimitLength(env, 60000)
    return env


if __name__ == '__main__':

    env = get_gym_env("ALE/Breakout-v5", render_mode="human")

    observation = env.reset()

    for _ in range(1000):
        observation, reward, done, info = env.step(env.action_space.sample())

        if done:
            observation = env.reset()

    env.close()
