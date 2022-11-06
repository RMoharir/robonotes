"""
This file contains implementation for the music theory reward function.
"""
from typing import List


NEUTRAL = 0
OCTAVE_STEPS = 12
OCTAVE_REWARD = 1
OCTAVE_PENALTY = -5
REPEAT_PENALTY = -5
KEY_REWARD = 1
KEY_PENALTY = -5
KEY_STEPS = [0, 2, 4, 5, 7, 9, 11]


def calc_reward(observation: List):
    """
    Engineered reward func from music theory:
    1) Notes be in same key. Key is determined based on first N notes.
    2) Note within an octave of previous note.
    3) Notes not repeated more than K times.

    :param observation: List of notes
    :return:
    """
    key_reward = get_key_reward(observation)
    octave_reward = get_octave_reward(observation)
    repeat_reward = get_repeat_reward(observation)
    # TODO can weight each type of reward
    total_reward = key_reward + octave_reward + repeat_reward
    info = {
        "key_reward": key_reward,
        "octave_reward": octave_reward,
        "repeat_reward": repeat_reward
    }
    return total_reward, info


def get_repeat_reward(observation: List) -> float:
    """
    Penalize if too many repeating notes.

    :param observation:
    :return:
    """
    if len(observation) < 2:
        return NEUTRAL

    current_note = observation[-1]
    prev_note = observation[-2]
    if current_note == prev_note and current_note > 1:
        return REPEAT_PENALTY
    return NEUTRAL


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
    #print(observation)
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
