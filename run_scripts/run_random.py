"""
Runs the random agent.

Example usage:
    python ./run_random.py --save_midi --max_trajectory_len 20
"""
from env import RoboNotesEnv
import argparse
from utils import plot_performance

from stable_baselines3.common.env_checker import check_env

MIDI_SAVEDIR = "./samples/random/"


def run(args):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    env.reset()

    check_env(env)

    plot = []
    for _ in range(args.num_trajectories):
        terminated = False
        while not terminated:
            action = env.action_space.sample()

            state, reward, terminated, info = env.step(action)
            plot.append(reward)
            if terminated:
                env.render()
                env.reset()
    if args.show_plot:
        plot_performance(plot)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_midi", action="store_true", help="Whether or not to save MIDI output")
    parser.add_argument("--show_plot", action="store_true", help="Whether or not to show performance plot")
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")
    parser.add_argument("--num_trajectories", type=int, default=1, required=False, help="Number of times to run agent")
    args = parser.parse_args()
    run(args)
