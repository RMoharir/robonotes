"""
Runs the random agent.

"""
from env import RoboNotesEnv


def run(num_trajectories=1, show_plot=False):

    env = RoboNotesEnv(max_trajectory_len=20, midi_savedir="tmp")

    for _ in range(num_trajectories):
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                env.render()
                env.reset()

    env.close()


if __name__ == "__main__":
    run()
