import argparse
import gym
import logging
import numpy as np
import shutil
import tensorflow as tf
import utils
import os
from ddpg_discrete import DDPG


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=1):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()

        done = False
        while not done:
            # たぶんbatch dimensionが必要
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(np.argmax(action))
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    tf.summary.scalar("eval_return", avg_reward,)
    return avg_reward


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")  # OpenAI gym environment name
    parser.add_argument("--env", default="CartPole-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e2,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=int)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--tau", default=0.001, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--logdir", default="./output/logdir", help="log directory")
    parser.add_argument("--parameter", default="./output/parameter", help="parameter directory")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--save_model_interval", default=int(1e4), type=int)
    args = parser.parse_args()

    shutil.rmtree(args.logdir, ignore_errors=True)

    env = gym.make(args.env)
    obs = env.reset()

    # Set seeds
    env.seed(args.seed)
    # tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_action = 1.0 #float(env.action_space.high[0])

    # Initialize policy
    if args.policy == "DDPG":
        policy = DDPG(state_dim, action_dim, max_action)
    else:
        raise ValueError("invalid policy {}".format(args.policy))

    replay_buffer = utils.ReplayBuffer()

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
    # restore model
    if args.model_dir is not None:
        assert os.path.isdir(args.model_dir)
        path_ckpt = tf.train.latest_checkpoint(args.model_dir)
        checkpoint.restore(path_ckpt)
        logger.info("Resotred {}".format(path_ckpt))

    obs = env.reset()

    total_timesteps = 0
    # summary_timesteps = tf.train.create_global_step()
    tf.summary.experimental.set_step(total_timesteps)

    while total_timesteps < args.max_timesteps:
        if total_timesteps > args.start_timesteps:
            if args.policy == "DDPG":
                total_actor_loss, total_critic_loss = policy.train(replay_buffer, batch_size=args.batch_size,
                                                                   tau=args.tau)
                tf.summary.scalar(name="ActorLoss", data=total_actor_loss, step=total_timesteps)
                tf.summary.scalar(name="CriticLoss", data=total_critic_loss, step=total_timesteps)
            else:
                raise ValueError("invalid policy {}".format(policy))

        if done:
            logger.info("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
            total_timesteps, episode_num, episode_timesteps, episode_reward))

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                cur_evaluate = evaluate_policy(policy)
                evaluations.append(cur_evaluate)
                checkpoint_manager.save()

                # tf.contrib.summary.scalar(name="online_return", tensor=cur_evaluate, step=total_timesteps,
                #                           family="loss")
                tf.summary.scalar(name="online_return", data=episode_reward, step=total_timesteps)
                writer.flush()

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if total_timesteps % args.save_model_interval == 0:
            checkpoint_manager.save()

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
            if action == 0:
                action = [1., 0.]
            else:
                action = [0., 1.]
        else:
            action = policy.select_action_noise(np.array(obs), noise_level=args.expl_noise)
            # if np.argmax(action) == 0:
            #     action = [1., 0.]
            # else:
            #     action = [0., 1.]
            # action = action.clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(np.argmax(action))

        # rendering
        env.render()

        # 上限ステップ由来のdoneであれば学習上はdoneとみなさない
        if episode_timesteps + 1 == env._max_episode_steps:
            done_bool = 0
        else:
            done_bool = float(done)

        episode_reward += reward

        # Store data in replay replay_buffer
        obs = obs.astype(np.float32)
        new_obs = new_obs.astype(np.float32)
        reward, action = np.float32(reward), np.int64(action)
        done_bool = np.array(done_bool, dtype=np.float32)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        tf.summary.experimental.set_step(total_timesteps)
        timesteps_since_eval += 1

    # Final evaluation
    # cur_evaluate = evaluate_policy(policy)
    # tf.contrib.summary.scalar(name="EvalReturn", tensor=cur_evaluate, step=total_timesteps)
    checkpoint_manager.save()

    tf.summary.flush()
