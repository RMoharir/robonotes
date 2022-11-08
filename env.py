from typing import Optional, Tuple
import time
import os

from gym import Env, spaces
from reward import calc_reward
from midi import convert_observation_to_midi_sequence
from midi import convert_to_midi_file
import numpy as np
from collections import defaultdict


class RoboNotesEnv(Env):
    """
    A music composition environment.
    """
    metadata = {
        "render_modes": ["human"]
    }

    def __init__(self, max_trajectory_len: int, midi_savedir=None):
        """

        :param max_trajectory_len: Max number of timesteps in the composition. Each ts is a 16th note.
        :param midi_savedir: If given, save MIDI file during render()
        """
        super(RoboNotesEnv, self).__init__()

        # choose from 36 pitches, with 2 special actions (0=no_event, 1=note_off)
        self.action_space = spaces.Discrete(38)
        # observation space is the sequence of notes of max_trajectory_len
        self.observation_space = spaces.Box(low=0, high=37, shape=(max_trajectory_len,), dtype=int)

        self.max_trajectory_len = max_trajectory_len
        self.midi_savedir = midi_savedir

        self.obs_trajectory = self.get_initial_ob()
        self.trajectory_idx = 0  # which index of self.obs_trajectory are we at
        self.total_reward = 0.0  # reward of self.obs_trajectory
        # for debugging
        self.partial_rewards = defaultdict(int)

    def get_initial_ob(self) -> np.ndarray:
        # action=0 means no_event
        return np.zeros(self.max_trajectory_len, dtype='int')

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> np.ndarray:
        super(RoboNotesEnv, self)

        self.obs_trajectory = self.get_initial_ob()
        self.total_reward = 0
        self.trajectory_idx = 0
        self.partial_rewards = defaultdict(int)
        return self.obs_trajectory

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.obs_trajectory[self.trajectory_idx] = action
        self.trajectory_idx += 1

        reward, info = calc_reward(self.obs_trajectory, self.trajectory_idx)
        self.total_reward += reward

        # check to see if end of song has been reached
        terminated = self.trajectory_idx >= self.max_trajectory_len
        # always False for now
        truncated = False
        for k, v in info.items():
            self.partial_rewards[k] += v
        return self.obs_trajectory, reward, terminated, info

    @staticmethod
    def save_midi(trajectory: np.ndarray, midi_savedir):
        render_trajectory = list(trajectory)
        midi_sequence = convert_observation_to_midi_sequence(render_trajectory)
        total_reward = 0
        total_info = {}
        for t in range(1, len(trajectory)):
            reward, info = calc_reward(trajectory, t)
            total_reward += reward
            for k in info:
                total_info[k] = total_info.get(k, 0) + info[k]

        print(f"Total reward: {total_reward}")
        print(f"Info: {total_info}")
        print(f"Trajectory: {render_trajectory}")
        print(f"MIDI sequence: {midi_sequence}")

        if midi_savedir:
            t = time.localtime()
            timestamp = time.strftime('%m-%d-%y_%H%M', t)
            fname = f"ts{timestamp}_ep{len(render_trajectory)}.midi"
            convert_to_midi_file(midi_sequence, os.path.join(midi_savedir, fname))

    def render(self, mode='human'):
        """
        We render the current state by printing the encoded actions and its corresponding MIDI sequence.
        TODO: can try to render and/or play MIDI file directly
        """
        render_trajectory = list(self.obs_trajectory)
        midi_sequence = convert_observation_to_midi_sequence(render_trajectory)
        print(f"Total reward: {self.total_reward}")
        print(f"Partial rewards: {self.partial_rewards}")
        print(f"Current state is: {render_trajectory}")
        print(f"MIDI sequence: {midi_sequence}")

        if self.midi_savedir:
            t = time.localtime()
            timestamp = time.strftime('%m-%d-%y_%H%M', t)
            fname = f"ts{timestamp}_ep{len(render_trajectory)}.midi"
            convert_to_midi_file(midi_sequence, os.path.join(self.midi_savedir, fname))
