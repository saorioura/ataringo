import numpy as np
import tensorflow as tf
from utils import PytorchInitializer, huber_loss, update_target_variables

layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses


class QFunc(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name="QFunc"):
        super().__init__(name=name)

        self.l1 = layers.Conv2D(32, 8, strides=4, padding="same", name="L1")
        self.l2 = layers.Conv2D(32, 4, strides=2, padding="same", name="L2")
        self.l3 = layers.Flatten()
        self.l4 = layers.Dense(128, kernel_initializer=PytorchInitializer, name="L4")
        self.l5 = layers.Dense(action_dim, kernel_initializer=PytorchInitializer, name="L5")

        # self.l1 = layers.Dense(32, kernel_initializer=PytorchInitializer(),
        #                        name="L1")
        # self.l2 = layers.Dense(32, kernel_initializer=PytorchInitializer(),
        #                        name="L2")
        # self.l3 = layers.Dense(action_dim, kernel_initializer=PytorchInitializer(),
        #                        name="L3")

        # 後段の処理のために早めにshapeを確定させる
        dummy_state = tf.constant(np.zeros(shape=np.insert(state_dim, 0, 1), dtype=np.float32))
        self(dummy_state)

    def call(self, inputs):
        with tf.device("/cpu:0"):
            features = tf.nn.relu(self.l1(inputs))
            features = tf.nn.relu(self.l2(features))
            features = self.l3(features)
            features = tf.nn.relu(self.l4(features))
            q_values = self.l5(features)

        return q_values


class DQN(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name="DQN", epsilon=0.5,):
        super().__init__()
        self.q_func = QFunc(state_dim, action_dim)
        self.q_func_target = QFunc(state_dim, action_dim)
        self.q_func_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # initialize target network
        for param, target_param in zip(self.q_func.weights, self.q_func_target.weights):
            target_param.assign(param)

        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 1e5
        self.target_replace_interval = int(1e2)
        self.n_update = 0
        self._action_dim = action_dim
        self.epsilon_decay_rate = (epsilon - self.epsilon_min) / self.epsilon_decay


    def select_action(self, state):
        """
        :param np.ndarray state:
        :return:
        """
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 3

        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action_body(tf.constant(state))

        return action.numpy()[0]


    def select_action_noise(self, state):
        """
        :param np.ndarray state:
        :return:
        """
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 3

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self._action_dim)
            return action
        else:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            action = self._select_action_body(tf.constant(state))
            return action.numpy()[0]

    @tf.function
    def _select_action_body(self, state):
        """
        :param np.ndarray state:
        :return:
        """
        q_values = self.q_func(state)
        return tf.argmax(q_values, axis=1)

    def train(self, replay_buffer, batch_size=64, discount=0.99):
        batch = replay_buffer.sample(batch_size)
        idx = [x[0] for x in batch]
        state = np.array([x[1][0] for x in batch]).astype(np.float32)
        next_state = np.array([x[1][1] for x in batch]).astype(np.float32)
        action = np.array([x[1][2] for x in batch]).astype(np.float32)
        reward = np.array([x[1][3] for x in batch]).astype(np.float32)
        done = np.array([x[1][4] for x in batch]).astype(np.float32)

        # state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        # state = np.array(state, dtype=np.float32)
        # next_state = np.array(next_state, dtype=np.float32)
        # action = np.array(action, dtype=np.float32)
        # reward = np.array(reward, dtype=np.float32)
        # done = np.array(done, dtype=np.float32)
        not_done = 1 - done
        td_error, q_func_loss = self._train_body(state, next_state, action, reward, not_done, discount)

        # update target network
        if self.n_update % self.target_replace_interval == 0:
            update_target_variables(self.q_func_target.weights, self.q_func.weights, tau=0.9)

        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_rate)

        self.n_update += 1
        tf.summary.scalar(name="DQN/epsilon", data=self.epsilon)
        tf.summary.scalar(name="DQN/q_func_loss", data=q_func_loss)

        return td_error, q_func_loss

    @tf.function
    def _train_body(self, state, next_state, action, reward, not_done, discount):
        with tf.device("/cpu:0"):

            with tf.GradientTape() as tape:
                # action_val = tf.expand_dims(tf.argmax(action, axis=1), axis=1)
                # action_val = tf.cast(action_val, dtype=tf.int32)
                action = tf.cast(tf.expand_dims(action, axis=1), dtype=tf.int32)
                indices = tf.concat(values=[tf.expand_dims(tf.range(state.shape[0]), axis=1),
                                            action], axis=1)
                current_Q = tf.expand_dims(tf.gather_nd(self.q_func(state), indices),
                                           axis=1)
                target_Q = self.q_func(next_state)
                target_Q = reward + (not_done * discount * tf.reduce_max(target_Q, keepdims=True, axis=1))
                target_Q = tf.stop_gradient(target_Q)
                td_error = current_Q - target_Q

                q_func_loss = tf.reduce_mean(huber_loss(td_error, delta=2.))
                # q_func_loss = tf.reduce_mean(tf.square(td_error))

            q_func_grad = tape.gradient(q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(zip(q_func_grad, self.q_func.trainable_variables))

            return td_error, q_func_loss
