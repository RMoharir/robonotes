"""
Runs the random agent.

Example usage:
    python ./run_dqn.py --save_midi --max_trajectory_len 20
"""
from env import RoboNotesEnv
import argparse
from utils import plot_performance

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

MIDI_SAVEDIR = "./samples/dqn/"


def create_dqn_model(env, **params):
    model = DQN("DQNPolicy", env, **params)
    return model


def run_dqn(args):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    check_env(env)

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000, log_interval=10)

    plot = []

    for _ in range(args.num_trajectories):
        terminated = False
        state = env.reset()
        while not terminated:
            action, _states = model.predict(state)
            state, reward, terminated, truncated, info = env.step(action)
            plot.append(reward)
            env.render()

    if args.show_plot:
        plot_performance(plot)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train parameters
    parser.add_argument("--learning_rate", type=float, default=20, required=False,
                        help="Length of music composition (number of beats)")
    # env parameters
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")
    # other parameters
    parser.add_argument("--save_midi", action="store_true", help="Whether or not to save MIDI output")
    parser.add_argument("--show_plot", action="store_true", help="Whether or not to show performance plot")
    args = parser.parse_args()
    run_dqn(args)
