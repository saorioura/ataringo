import argparse
import gym
import logging
import numpy as np
import shutil
import tensorflow as tf
import utils
from dqn_cnn import DQN
import cv2
from collections import deque
from gym import spaces

class atari_wrapper(gym.Wrapper):
    def __init__(self, env, override_num_noops=30, skip=4, max_episode_steps=None):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.override_num_noops = override_num_noops
        self.noop_action = 0
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        if max_episode_steps is None:
            self._max_episode_steps = env.spec.max_episode_steps
        else:
            self._max_episode_steps = max_episode_steps

    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        noops = self.override_num_noops

        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        # return self.env.step(ac)
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class obs_wrapper(gym.ObservationWrapper):
    def __init__(self, env=None, max_episode_steps=None):
        super().__init__(env)
        self._width = 48
        self._height = 48
        new_space = gym.spaces.Box(low=0, high=1, shape=(self._height, self._width, 1), dtype=np.float32)
        original_space = self.observation_space
        self.observation_space = new_space
        if max_episode_steps is None:
            self._max_episode_steps = env._max_episode_steps
        else:
            self._max_episode_steps = max_episode_steps

    def observation(self, obs):
        frame = obs
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = frame[16:-16, 0:64]
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, -1)
        obs = frame / 255.
        return obs

class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See also baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=1):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()

        done = False
        while not done:
            # たぶんbatch dimensionが必要
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            # env.render()
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    tf.summary.scalar("eval_return", avg_reward)
    return avg_reward


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DQN")  # OpenAI gym environment name
    parser.add_argument("--env", default="Breakout-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=int)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.9, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    # parser.add_argument("--tau", default=1, type=float)  # Target network update rate
    # parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--logdir", default="./output/logdir", help="log directory")
    parser.add_argument("--parameter", default="./output/parameter", help="parameter directory")
    args = parser.parse_args()

    shutil.rmtree(args.logdir, ignore_errors=True)

    env = gym.make(args.env)
    env = atari_wrapper(env, override_num_noops=1, skip=4)
    env = obs_wrapper(env)
    env = FrameStack(env, 4)
    obs = env.reset()

    # Set seeds
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # state_dim = env.observation_space.shape[0]
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # Initialize policy
    if args.policy == "SAC":
        policy = SAC.SAC(state_dim, action_dim, max_action)
    elif args.policy == "DDPG":
        policy = DDPG(state_dim, action_dim, max_action)
    elif args.policy == "DQN":
        policy = DQN(state_dim, action_dim)
    else:
        raise ValueError("invalid policy {}".format(args.policy))

    replay_buffer = utils.PrioritizedReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]

    timesteps_since_eval = 0
    episode_num = 0
    done = False
    episode_reward = 0
    episode_timesteps = 0

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                                 directory=args.parameter,
                                                                 max_to_keep=5)

    writer = tf.summary.create_file_writer(args.logdir)
    writer.set_as_default()
    # tf.contrib.summary.initialize()

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    obs = env.reset()

    total_timesteps = tf.Variable(0, dtype=tf.int64)
    tf.summary.experimental.set_step(total_timesteps)
    # summary_timesteps = tf.train.create_global_step()

    while total_timesteps.numpy() < args.max_timesteps:
        if total_timesteps.numpy() > args.start_timesteps:
            if args.policy == "SAC":
                total_actor_loss, total_v_loss, total_q_loss = policy.train(replay_buffer,
                                                                            batch_size=args.batch_size)
                tf.summary.scalar(name="V_Loss", data=total_v_loss)
                tf.summary.scalar(name="Q_Loss", data=total_q_loss)
                tf.summary.scalar(name="ActorLoss", data=total_actor_loss)
            elif args.policy == "DDPG":
                total_actor_loss, total_critic_loss = policy.train(replay_buffer, batch_size=args.batch_size)
                tf.summary.scalar(name="ActorLoss", data=total_actor_losss)
                tf.summary.scalar(name="CriticLoss", data=total_critic_loss)
            elif args.policy == "DQN":
                td_errors, q_func_loss = policy.train(replay_buffer, batch_size=args.batch_size)
                # tf.summary.scalar(name="q_func_loss", data=q_func_loss)
            else:
                raise ValueError("invalid policy {}".format(policy))

        if done:
            logger.info("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
            total_timesteps.numpy(), episode_num, episode_timesteps, episode_reward))
            tf.summary.scalar(name="episode_rward", data=episode_reward)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                cur_evaluate = evaluate_policy(policy)
                evaluations.append(cur_evaluate)
                checkpoint_manager.save()

                tf.summary.scalar(name="online_return", data=cur_evaluate)
                writer.flush()

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps.numpy() < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action_noise(np.array(obs))

        # rendering
        # env.render()

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        # print(episode_num, episode_timesteps, action, reward)

        cv2.imshow('Breakout', new_obs[0])
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

        # 上限ステップ由来のdoneであれば学習上はdoneとみなさない
        if episode_timesteps + 1 == env._max_episode_steps:
            done_bool = 0
        else:
            done_bool = float(done)

        episode_reward += reward

        # Store data in replay replay_buffer
        obs = np.float32(obs)
        new_obs = np.float32(new_obs)
        reward, action = np.float32(reward), np.int64(action)
        done_bool = np.array(done_bool, dtype=np.float32)
        replay_buffer.add(1, (obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps.assign_add(1)
        timesteps_since_eval += 1

    # Final evaluation
    cur_evaluate = evaluate_policy(policy)
    tf.summary.scalar(name="EvalReturn", data=cur_evaluate, step=total_timesteps)
    checkpoint_manager.save()

    tf.summary.flush()
