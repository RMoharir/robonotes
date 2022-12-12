"""
Runs the random agent.

Example usage:
    python ./run_scripts/run_ppo.py --exp_name ppo --total_timesteps 2000000 --save_model
    python ./run_scripts/run_ppo.py --exp_name ppo_multi2 --total_timesteps 2000000 --num_pitches 3 --save_model --save_midi
    python ./run_scripts/run_ppo.py --exp_name ppo --load_model_path run_logs/ppo_multi_11-12-2022_13-04-29/ppo.zip \
        --num_pitches 2 --save_midi
"""
import os
import time

from env import RoboNotesEnv
from complex_env import RoboNotesComplexEnv
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

MIDI_SAVEDIR = "./samples/ppo/"


def create_ppo_model(env, args, log_dir=None):
    model = PPO("MlpPolicy", env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_advantage=args.normalize_advantage,
                tensorboard_log=log_dir)

    return model


def train_loop(model, timesteps, log_interval, test_interval, progress_bar=True):
    #TODO: add EvalCallback to use test_interval
    model.learn(total_timesteps=timesteps, log_interval=log_interval, progress_bar=progress_bar)


def sample_test_trajectory(model, env, num_trajectories=3):
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
            obs, _ = env.reset()
        time.sleep(2)


def run_ppo(args, log_dir=None):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    if args.num_pitches > 1:
        env = RoboNotesComplexEnv(max_trajectory_len=args.max_trajectory_len,
                                  num_pitches=args.num_pitches,
                                  midi_savedir=midi_savedir)
    else:
        env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    check_env(env)

    model = create_ppo_model(env, args, log_dir=log_dir)

    if args.load_model_path:
        print(f"Skipping training. Loading model from path: {args.load_model_path}")
        model = model.load(args.load_model_path)
    else:
        print(f"Start training.")
        train_loop(
            model,
            timesteps=args.total_timesteps,
            test_interval=args.test_interval,
            log_interval=args.log_interval
        )
        if args.save_model:
            model.save(f"{log_dir}/ppo")

    print("Finished. Sampling test trajectories.")
    sample_test_trajectory(model, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="name the experiment, for log dir")

    # train parameters
    parser.add_argument("--total_timesteps", type=int, default=10000002, required=False,
                        help="number of timesteps (env steps) to train")
    parser.add_argument("--log_interval", type=int, default=1, required=False,
                        help="number of timesteps before logging")
    parser.add_argument("--test_interval", type=int, default=100000, required=False,
                        help="number of timesteps before testing")
    parser.add_argument("--learning_rate", type=float, default=0.0003, required=False,
                        help="learning rate")
    parser.add_argument("--batch_size", type=float, default=64, required=False,
                        help="minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10, required=False,
                        help="number of epoch when optimizing the surrogate loss")
    parser.add_argument("--gamma", type=float, default=0.99, required=False,
                        help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, required=False,
                        help="factor for trade-off of bias vs variance for Generalized Advantage Estimator")
    parser.add_argument("--n_steps", type=int, default=2048, required=False,
                        help="number of steps to run for each environment per update")
    parser.add_argument("--normalize_advantage", type=bool, default=True, required=False,
                        help="Whether to normalize or not the advantage")
    # env parameters
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")
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

    run_ppo(args, log_dir=logdir)
