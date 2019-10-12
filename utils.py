import numpy as np
import tensorflow as tf

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

NUMPY_DTYPE = np.float32
TF_DTYPE = tf.float32

# Simple replay replay_buffer
class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


class PytorchInitializer(tf.keras.initializers.Initializer):
    """PytorchのLinearにあわせたinitializer
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    :param seed:
    :return:
    """
    def __init__(self, scale=1.0, seed=None):
        self.seed = seed
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = TF_DTYPE

        if len(shape) == 1:
            fan_in = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
        else:
            raise ValueError("invalid shape")

        scale = self.scale * fan_in

        stdv = 1. / tf.math.sqrt(scale)

        return tf.random.uniform(
            shape, -stdv, stdv, dtype=dtype, seed=self.seed)

def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):
    """
    Returns an op to update a list of target variables from source variables.
    The update rule is:
    `target_variable = (1 - tau) * target_variable + tau * source_variable`.
    :param target_variables: a list of the variables to be updated.
    :param source_variables: a list of the variables used for the update.
    :param tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
        with small tau representing an incremental update, and tau == 1
        representing a full update (that is, a straight copy).
    :param use_locking: use `tf.Variable.assign`'s locking option when assigning
        source variable values to target variables.
    :param name: sets the `name_scope` for this op.
    :raise TypeError: when tau is not a Python float
    :raise ValueError: when tau is out of range, or the source and target variables
        have different numbers or shapes.
    :return: An op that executes all the variable updates.
    """
    if not isinstance(tau, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(tau))
    if not 0.0 < tau <= 1.0:
        raise ValueError("Invalid parameter tau {}".format(tau))
    if len(target_variables) != len(source_variables):
        raise ValueError("Number of target variables {} is not the same as "
                         "number of source variables {}".format(
                             len(target_variables), len(source_variables)))

    same_shape = all(trg.get_shape() == src.get_shape()
                     for trg, src in zip(target_variables, source_variables))
    if not same_shape:
        raise ValueError("Target variables don't have the same shape as source "
                         "variables.")

    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable, use_locking)
        else:
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable, use_locking)

    # with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer(object):   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity=1000001):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)