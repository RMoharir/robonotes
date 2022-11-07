"""
Runs the random agent.

Example usage:
    python ./run_scripts/run_dqn.py --exp_name first_test --save_midi
"""
import os
import time

from env import RoboNotesEnv
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

MIDI_SAVEDIR = "./samples/dqn/"


def create_dqn_model(env, args, log_dir):
    model = DQN("MlpPolicy", env,
                learning_rate=args.learning_rate,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                exploration_fraction=0.1,
                target_update_interval=args.target_update_interval,
                tensorboard_log=log_dir)

    return model


def train_loop(model, timesteps, log_interval, progress_bar=True):
    model.learn(total_timesteps=timesteps, log_interval=log_interval, progress_bar=progress_bar)


def sample_test_trajectory(model, env, num_trajectories=20, midi_savedir=None):
    """
    Test the model by sampling trajectories and render() after each one (saves MIDI if path provided)
    :param model:
    :param num_trajectories:
    :return: List of trajectories and List of their total rewards
    """
    obs, _ = env.reset()

    for _ in range(num_trajectories):
        terminated = False
        while not terminated:
            action, _states = model.predict(obs)
            state, reward, terminated, _, info = env.step(action)
        if terminated:
            env.render()
            env.reset()
        # final_trajectory = env.buf_infos[0]['terminal_observation']
        # RoboNotesEnv.save_midi(final_trajectory, midi_savedir)


def run_dqn(args, log_dir):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None
    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    check_env(env)
    model = create_dqn_model(env, args, log_dir)

    if args.load_model_path:
        print(f"Loading model from path: {args.load_model_path}")
        model = model.load(args.load_model_path)
    else:
        train_loop(model, timesteps=args.total_timesteps, log_interval=args.log_interval)
        if args.save_model:
            model.save(f"{log_dir}/dqn")

    sample_test_trajectory(model, env, midi_savedir=midi_savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="name the experiment, for logging")
    # train parameters
    parser.add_argument("--total_timesteps", type=int, default=500000, required=False,
                        help="number of timesteps (env steps) to train")
    parser.add_argument("--log_interval", type=int, default=20, required=False,
                        help="number of timesteps before logging")
    # model parameters
    parser.add_argument("--learning_rate", type=float, default=0.0001, required=False,
                        help="learning rate")
    parser.add_argument("--learning_starts", type=int, default=100000, required=False,
                        help="how many steps of the model to collect transitions for before learning starts")
    parser.add_argument("--batch_size", type=int, default=64, required=False,
                        help="minibatch size of each GD step")
    parser.add_argument("--train_freq", type=int, default=20, required=False,
                        help="update model every X steps")
    parser.add_argument("--gradient_steps", type=int, default=1, required=False,
                        help="how many gradient steps to do after each rollout")
    parser.add_argument("--target_update_interval", type=int, default=10, required=False,
                        help="update target network every X env steps")

    # env parameters
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")
    # other parameters
    parser.add_argument("--save_midi", action="store_true", help="Whether or not to save MIDI output")
    parser.add_argument("--save_model", action="store_true", help="Whether or not to save trained dqn model")
    parser.add_argument("--load_model_path", type=str,
                        help="If provided, path of the dqn model to load.", required=False)
    args = parser.parse_args()
    print("Args: ", args)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../run_logs')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    run_dqn(args, logdir)
