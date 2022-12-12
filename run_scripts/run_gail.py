"""
Runs the random agent.

Example usage:
    python ./run_scripts/run_gail.py --exp_name gail --save_model
    python ./run_scripts/run_gail.py --exp_name gail --load_model_path run_logs/gail_12-12-2022_01-03-34/gail.zip --save_midi
"""
import os
import time

from env import RoboNotesEnv
import argparse

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo import PPO
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.rollout import flatten_trajectories
from midi.sampler import RoboNotesSampler
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.vec_env import DummyVecEnv

MIDI_SAVEDIR = "./samples/gail/"
MIDI_FILEPATH = "data/midi_arrays.npy"


def create_gail_models(env, expert_rollouts, log_dir=None):
    learner = PPO("MlpPolicy", env,
                  learning_rate=args.learning_rate,
                  n_steps=args.n_steps,
                  batch_size=args.batch_size,
                  n_epochs=args.n_epochs,
                  gamma=args.gamma,
                  gae_lambda=args.gae_lambda,
                  normalize_advantage=args.normalize_advantage,
                  tensorboard_log=log_dir)

    reward_net = BasicRewardNet(
        env.observation_space,
        env.action_space,
        normalize_input_layer=RunningNorm,
    )

    venv = DummyVecEnv([lambda: RoboNotesEnv(max_trajectory_len=args.max_trajectory_len)])
    trainer = GAIL(
        demonstrations=flatten_trajectories(expert_rollouts),
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=10,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        log_dir=logdir
    )

    return trainer, learner


def train_loop(model, total_timesteps, log_interval, progress_bar=True):
    model.train(
        total_timesteps=total_timesteps,
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
        rewards.append(rew)
        time.sleep(2)
    print(rewards)


def run_gail(args, log_dir=None):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    check_env(env)

    sampler = RoboNotesSampler(midi_file=MIDI_FILEPATH)

    expert_rollouts = sampler.sample_trajectories(num_trajectories=args.num_trajectories,
                                                  max_trajectory_len=args.max_trajectory_len)
    trainer, learner = create_gail_models(env, expert_rollouts, log_dir=log_dir)

    if args.load_model_path:
        print(f"Skipping training. Loading model from path: {args.load_model_path}")
        learner = learner.load(args.load_model_path)
    else:
        print(f"Start training.")
        train_loop(
            trainer,
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval
        )
        if args.save_model:
            learner.save(f"{log_dir}/gail")

    print("Finished. Sampling test trajectories.")
    sample_test_trajectory(learner, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="name the experiment, for log dir")

    # train parameters
    parser.add_argument("--total_timesteps", type=int, default=500000, required=False,
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
    parser.add_argument("--n_steps", type=int, default=1024, required=False,
                        help="number of steps to run for each environment per update")
    parser.add_argument("--normalize_advantage", type=bool, default=True, required=False,
                        help="Whether to normalize or not the advantage")
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

    run_gail(args, log_dir=logdir)
