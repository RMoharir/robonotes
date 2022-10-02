"""
This file contains implementation for the music theory reward function.
"""
from typing import List


def calc_reward(observation: List):
    """
    Engineered reward func given some music theory rules:
    1) Notes be in same key.
    2) Note within an octave of previous note.
    3) Notes not repeated.

    :param observation:
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
    return -1


def get_octave_reward(observation: List) -> float:
    """
    Return reward for consecutive notes being in the same octave.
    :param observation:
    :return:
    """
    return -1


def get_key_reward(observation: List) -> float:
    """
    We do the simplest thing and return first note as the key.
    :param observation:
    :return:
    """
    key_reward = 0
    for note in observation:
        # if note is in the same key, add
        # if note in different key, penalize
        pass
    return key_reward
