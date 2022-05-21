import tensorflow as tf
from umr.utils import get_gym_env
from umr.a3c.model import A3C
import numpy as np

env_name = "Breakout-v5"

env = get_gym_env(f"ALE/{env_name}", render_mode=None, is_train=False)

model = A3C(env.action_space.n)
latest = tf.train.latest_checkpoint(f'./train_log/train-ALE/{env_name}')
model.load_weights(latest)
observation = env.reset()
score = 0
scores = []
for _ in range(50):
    while True:
        logits, _ = model(observation[np.newaxis, :])
        distrib = tf.nn.softmax(logits)
        if np.random.random(1)[0] < 0.01:
            action = np.random.choice(env.action_space.n, p=distrib.numpy()[0])
        else:
            action = np.argmax(distrib.numpy()[0])
        observation, reward, done, info = env.step(action)
        # env.render("human")
        score += reward
        if done:
            print(score)
            scores.append(score)
            observation = env.reset()
            score = 0
            break
env.close()

print(np.mean(scores))