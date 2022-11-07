"""
This file contains implementation for the music theory reward function.
"""
from typing import List

import numpy as np

# consts
NEUTRAL = 0

# same octave
OCTAVE_STEPS = 12
OCTAVE_REWARD = 5
OCTAVE_PENALTY = -5

# no repeating notes
REPEAT_REWARD = 10
REPEAT_PENALTY = -50

# same key
KEY_REWARD = 10
KEY_PENALTY = -20
KEY_STEPS = [0, 2, 4, 5, 7, 9, 11]

# not empty
FIRST_NOTE_PENALTY = -100
EMPTY_PENALTY = -10


def calc_reward(observation: np.ndarray, trajectory_idx: int):
    """
    Engineered reward func from music theory:

    1) Notes be in same key. Key is determined based on first N notes
    2) Note within an octave of previous note
    3) Notes should not repeat too many times
    4) Trajectory should not be empty

    :param observation:
    :param trajectory_idx: curr idx of the trajectory (only calculate reward for trajectory up to idx)
    :return:
    """
    assert trajectory_idx > 0
    trajectory = list(observation)[:trajectory_idx]

    key_reward = get_key_reward(trajectory)
    octave_reward = get_octave_reward(trajectory)
    repeat_penalty = get_repeat_penalty(trajectory)
    empty_penalty = get_empty_penalty(trajectory)

    # TODO can weight each type of reward
    total_reward = key_reward + octave_reward + repeat_penalty + empty_penalty
    info = {
        "key_reward": key_reward,
        "octave_reward": octave_reward,
        "repeat_penalty": repeat_penalty,
        "empty_penalty": empty_penalty,
    }
    return total_reward, info


def get_empty_penalty(observation: List) -> float:
    if len(observation) == 1 and observation[0] in {0, 1}:
        return FIRST_NOTE_PENALTY

    if observation[-1] in {0, 1}:
        return EMPTY_PENALTY

    return NEUTRAL


def get_repeat_penalty(observation: List) -> float:
    """
    Penalize if too many repeating notes.

    :param observation:
    :return:
    """
    if len(observation) < 2:
        return NEUTRAL

    current_note = observation[-1]
    prev_note = observation[-2]
    if current_note == prev_note:
        return REPEAT_PENALTY

    return REPEAT_REWARD


def get_octave_reward(observation: List) -> float:
    """
    Return reward for consecutive notes being in the same octave.
    :param observation:
    :return:
    """
    if len(observation) < 2:
        return NEUTRAL
    current_note = observation[-1]
    prev_note = observation[-2]
    if abs(current_note - prev_note) <= OCTAVE_STEPS:
        return OCTAVE_REWARD
    # penalize if not in same octave
    return OCTAVE_PENALTY


def get_key_reward(observation: List) -> float:
    """
    We do the simplest thing and return first note as the key.
    :param observation:
    :return:
    """
    def is_in_key(ob, key):
        if abs(ob - key) % OCTAVE_STEPS in KEY_STEPS:
            return True
        return False

    if len(observation) < 2:
        return NEUTRAL
    
    key = [ob for ob in observation][0]

    current_note = observation[-1]
    if is_in_key(current_note, key):
        return KEY_REWARD
    return KEY_PENALTY
