from matplotlib import pyplot as plt


def plot_performance(rewards_plot):
    """
    Plots the performance given list of rewards.
    :return:
    """
    plt.plot(rewards_plot)
    plt.title("Rewards vs. Trajectory Timesteps")
    plt.show()
