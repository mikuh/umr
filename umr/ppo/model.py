import tensorflow as tf


class PPO(tf.keras.Model):

    def __init__(self, action_size: int, image_size=(84, 84), frame_history=4):
        super(PPO, self).__init__()
        self.reshape_size = (-1, *image_size, 3 * frame_history)
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')
        self.maxp1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')
        self.maxp2 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')
        self.maxp3 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(512)
        self.prelu = tf.keras.layers.PReLU()

        self.logits = tf.keras.layers.Dense(action_size)
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        assert state.shape.rank == 5  # Batch, H, W, Channel, History
        state = tf.transpose(state, [0, 1, 2, 4, 3])  # swap channel & history
        state = tf.reshape(state, self.reshape_size)
        x = self.conv1(state)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.prelu(x)

        logits = self.logits(x)
        value = self.value(x)

        return logits, value


def create_model(action_size: int, image_size=(84, 84), channel=3, frame_history=4):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(*image_size, channel, frame_history))
    x = tf.transpose(inputs, [0, 1, 2, 4, 3])
    x = tf.reshape(x, (-1, *image_size, channel * frame_history))
    x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.PReLU()(x)

    logits = tf.keras.layers.Dense(action_size)(x)
    value = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=(logits, value), name="mnist_model")

    return model
