"""
Runs the random agent.

Example usage:
    python ./run_scripts/run_bc.py --exp_name bc --n_epochs 10 --save_model
    python ./run_scripts/run_bc.py --exp_name bc --n_epochs 10 --load_model_path run_logs/bc_11-12-2022_23-29-58/bc --save_midi
"""
import os
import time

from env import RoboNotesEnv
import argparse

from stable_baselines3.common.env_checker import check_env
from imitation.algorithms import bc
from imitation.data.rollout import flatten_trajectories
from midi.sampler import RoboNotesSampler
import numpy as np

MIDI_SAVEDIR = "./samples/bc/"
MIDI_FILEPATH = "data/midi_arrays.npy"


def create_bc_model(env, expert_rollouts):
    rng = np.random.default_rng(0)

    model = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=flatten_trajectories(expert_rollouts),
        rng=rng,
    )

    return model


def train_loop(model, n_epochs, log_interval, progress_bar=True):
    model.train(
        n_epochs=n_epochs,
        log_interval=log_interval,
        progress_bar=progress_bar
    )


def sample_test_trajectory(model, env, num_trajectories=3):
    """
    Test the model by sampling trajectories and render() after each one (saves MIDI if path provided)
    :param model:
    :param num_trajectories:
    :return: List of trajectories and List of their total rewards
    """
    obs, _ = env.reset()

    rewards = []
    for _ in range(num_trajectories):
        rew = 0
        terminated = False
        while not terminated:
            action, _ = model.predict(obs)
            state, reward, terminated, _, info = env.step(action)
            rew += reward
        if terminated:
            env.render()
            obs, _ = env.reset()
        time.sleep(2)
    print(rewards)


def run_bc(args, log_dir=None):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    check_env(env)

    sampler = RoboNotesSampler(midi_file=MIDI_FILEPATH)

    expert_rollouts = sampler.sample_trajectories(num_trajectories=args.num_trajectories,
                                                  max_trajectory_len=args.max_trajectory_len)
    model = create_bc_model(env, expert_rollouts)

    if args.load_model_path:
        print(f"Skipping training. Loading model from path: {args.load_model_path}")
        model = bc.reconstruct_policy(args.load_model_path)
        print("Finished. Sampling test trajectories.")
        sample_test_trajectory(model, env)
    else:
        print(f"Start training.")
        train_loop(
            model,
            n_epochs=args.n_epochs,
            log_interval=args.log_interval
        )
        if args.save_model:
            model.save_policy(f"{log_dir}/bc")

        print("Finished. Sampling test trajectories.")
        sample_test_trajectory(model.policy, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="name the experiment, for log dir")

    # train parameters
    parser.add_argument("--log_interval", type=int, default=1, required=False,
                        help="number of timesteps before logging")
    parser.add_argument("--n_epochs", type=int, default=10, required=False,
                        help="number of epoch when optimizing the surrogate loss")
    # env parameters
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")
    parser.add_argument("--num_trajectories", type=int, default=10000, required=False,
                        help="Number of trajectories to train BC Agent")
    parser.add_argument("--num_pitches", type=int, default=1, required=False,
                        help="Number of pitches.")
    # other parameters
    parser.add_argument("--save_midi", action="store_true", help="Whether or not to save MIDI output")
    parser.add_argument("--save_model", action="store_true", help="Whether or not to save trained dqn model")
    parser.add_argument("--load_model_path", type=str,
                        help="If provided, path of the dqn model to load.", required=False)
    args = parser.parse_args()
    print("\nARGS: ", args, "\n")

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../run_logs')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    if args.load_model_path:
        logdir = None
    else:
        logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)

    print("\nLOGGING TO: ", logdir, "\n")

    run_bc(args, log_dir=logdir)
