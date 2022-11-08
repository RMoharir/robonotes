"""
Runs the random agent.

Example usage:
    python ./run_scripts/run_random.py --exp_name=random --total_timesteps 1000000
"""
from env import RoboNotesEnv
import argparse
import os
import time
import tensorflow as tf

from stable_baselines3.common.env_checker import check_env

MIDI_SAVEDIR = "./samples/random/"


def run_random(args, log_dir=None):
    summary_writer = tf.summary.create_file_writer(log_dir)
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    env.reset()
    check_env(env)

    ts = 0
    rollouts = 0
    while ts < args.total_timesteps:
        terminated = False
        while not terminated:
            action = env.action_space.sample()

            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                with summary_writer.as_default():
                    tf.summary.scalar("rollout/ep_rew_mean", env.total_reward, step=ts)
                env.reset()
            ts += 1
        if (rollouts+1) % 10000 == 0:
            print(f"rollouts={rollouts}, ts={ts}/{args.total_timesteps}")
        rollouts += 1

    env.reset()
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="name the experiment, for log dir")

    # train parameters
    parser.add_argument("--total_timesteps", type=int, default=10000000, required=False,
                        help="number of timesteps (env steps) to train")
    parser.add_argument("--log_interval", type=int, default=100, required=False,
                        help="number of timesteps before logging")
    parser.add_argument("--test_interval", type=int, default=100000, required=False,
                        help="number of timesteps before testing")

    # env parameters
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")
    # other parameters
    parser.add_argument("--save_midi", action="store_true", help="Whether or not to save MIDI output")
    args = parser.parse_args()
    print("\n\n\nARGS: ", args, "\n\n\n")

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../run_logs')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    run_random(args, log_dir=logdir)
