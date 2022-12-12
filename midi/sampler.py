from typing import List
import numpy as np
import random
from imitation.data.types import Trajectory


class RoboNotesSampler(object):
    def __init__(self, midi_file: str, seed=0):
        self.midi_seqs = np.load(midi_file, allow_pickle=True)

        print(f"Loaded {len(self.midi_seqs)} midi tracks.")
        random.seed(seed)

    def sample_trajectories(self, num_trajectories: int, max_trajectory_len: int) -> List[Trajectory]:
        trajectories = []
        for _ in range(num_trajectories):
            trajectories.append(self.sample_single_trajectory(max_trajectory_len=max_trajectory_len))
        return trajectories

    def sample_single_trajectory(self, max_trajectory_len: int) -> Trajectory:
        midi_seq = random.choice(self.midi_seqs)
        while len(midi_seq) < max_trajectory_len:
            midi_seq = random.choice(self.midi_seqs)
        start = random.randint(0, len(midi_seq) - max_trajectory_len)

        selected_seq = midi_seq[start:start + max_trajectory_len]
        acts = np.array(selected_seq)
        obs = [np.zeros((max_trajectory_len, ))]
        for i, a in enumerate(acts):
            ob = np.copy(obs[-1])
            ob[i] = a
            obs.append(ob)
        return Trajectory(obs=np.array(obs), acts=acts, infos=None, terminal=True)
