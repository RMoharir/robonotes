"""
Runs the random agent.

Example usage:
    python ./run_ppo.py --save_midi --max_trajectory_len 20
"""
from env import RoboNotesEnv
import argparse
from utils import plot_performance

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

MIDI_SAVEDIR = "./samples/random/"


def run_ppo(args):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None

    env = DummyVecEnv([lambda: RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)])
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo")

    plot = []

    for _ in range(args.num_trajectories):
        terminated = False
        state = env.reset()
        while not terminated:
            action, _states = model.predict(state)

            state, reward, terminated, info = env.step(action)

            plot.append(reward)

            env.render()

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
    run_ppo(args)
