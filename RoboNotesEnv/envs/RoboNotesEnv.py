from typing import Optional, Tuple, List
import time
import os

from gym import Env, spaces
from reward import calc_reward
from midi import convert_observation_to_midi_sequence
from midi import convert_to_midi_file
import numpy as np

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

        self.action_space = spaces.Discrete(38)  # choose from 36 pitches, with 2 special actions (note_off, no_event)
        self.observation_space = spaces.Discrete(38)  # obv is the sequence of notes so far

        self.state = 0
        self.observations = []
        self.collected_reward = 0

        self.max_trajectory_len = max_trajectory_len
        self.midi_savedir = midi_savedir

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[List, dict]:
        info = {}
        self.state = 0
        self.observations = []
        self.collected_reward = 0
        return self.state

    def step(self, action: int):
        assert self.action_space.contains(action)
        truncated = False

        self.state = action
        self.observations.append(action)
        reward, info = calc_reward(self.observations)
        self.collected_reward += reward

        # check to see if end of song has been reached
        terminated = len(self.observations) >= self.max_trajectory_len
        return self.state, reward, terminated, info

    def render(self):
        """
        We render the current state by printing the encoded actions and its corresponding MIDI sequence.
        TODO: can try to render and/or play MIDI file directly
        """
        midi_sequence = convert_observation_to_midi_sequence(self.observations)
        print(f"Total reward: {self.collected_reward}")
        print(f"Current state is: {self.state}")
        print(f"MIDI sequence: {midi_sequence}")

        if self.midi_savedir:
            t = time.localtime()
            timestamp = time.strftime('%m-%d-%y_%H%M', t)
            fname = f"ts{timestamp}_ep{len(self.observations)}.midi"
            convert_to_midi_file(midi_sequence, os.path.join(self.midi_savedir, fname))
