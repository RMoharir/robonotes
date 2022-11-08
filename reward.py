"""
This file contains implementation for the music theory reward function.
"""
from typing import List
from typing import Dict
from typing import Tuple

import numpy as np
import statsmodels.api as sm
from scipy.stats import entropy


# consts
NEUTRAL = 0

# penalize jump by more than an octave
OCTAVE_STEPS = 12
OCTAVE_PENALTY = -1

# penalize repeating notes
REPEAT_PENALTY = -1

# reward same key
END_ON_TONIC_REWARD = 5
KEY_REWARD = 5
KEY_STEPS = [0, 2, 4, 5, 7, 9, 11]

# penalize if empty
EMPTY_PENALTY = -2
EMPTY_ON_FIRST = -100
EMPTY_NOTES = {0, 1}

# reward diversity (correlation < 0.15)
# DIVERSITY_REWARD = 5
DIVERSITY_PENALTY = -2


def calc_reward(observation: np.ndarray, trajectory_idx: int) -> Tuple[float, Dict]:
    """
    Hand engineered reward based on music theory:

    1) Reward notes in same key. Key is determined to be the first note in composition.
    2) Penalize if note is greater than an octave of previous note.
    3) Penalize if same note repeat too many times in a row.
    4) Penalize empty notes.
    5) Reward diversity in composition

    :param observation: of length max_trajectory_len
    :param trajectory_idx: curr idx of the trajectory (only calculate reward for trajectory up to idx)
    :return:
    """
    assert trajectory_idx > 0
    observation = list(observation)
    trajectory = observation[:trajectory_idx]

    key_reward = get_key_reward(trajectory, is_terminal=(trajectory_idx == len(observation)))
    octave_penalty = get_octave_penalty(trajectory)
    repeat_penalty = get_repeat_penalty(trajectory)
    empty_penalty = get_empty_penalty(trajectory)
    diversity_reward = get_diversity_reward(trajectory)

    # TODO can weight each type of reward
    total_reward = key_reward + octave_penalty + repeat_penalty + empty_penalty + diversity_reward
    info = {
        "key_reward": key_reward,
        "octave_penalty": octave_penalty,
        "repeat_penalty": repeat_penalty,
        "empty_penalty": empty_penalty,
        "diversity_reward": diversity_reward
    }
    return total_reward, info


def get_empty_penalty(observation: List) -> float:
    """
    Penalize if last note(s) are empty.
    The longer the sequence of empty notes, the higher the penalty.
    """
    # super discourage empty on first note
    if len(observation) == 1 and observation[0] in EMPTY_NOTES:
        return EMPTY_ON_FIRST
    # penalty increases with consecutive empty notes from last note
    num_empty = 0
    for ob in reversed(observation):
        if ob in EMPTY_NOTES:
            num_empty += 1
        else:
            break
    return num_empty * EMPTY_PENALTY


def get_repeat_penalty(observation: List) -> float:
    """
    Penalize repeated notes of more than three.
    """
    current_note = observation[-1]
    num_repeat = 0
    for ob in reversed(observation):
        if ob == current_note:
            num_repeat += 1
        else:
            break
    num_repeat -= 1
    assert num_repeat >= 0
    return num_repeat * REPEAT_PENALTY


def get_octave_penalty(observation: List) -> float:
    """
    Penalize if note jumped by more than one octave.
    """
    if len(observation) < 2:
        return NEUTRAL

    current_note = observation[-1]
    prev_note = observation[-2]
    if abs(current_note - prev_note) > OCTAVE_STEPS:
        return OCTAVE_PENALTY

    return NEUTRAL


def get_key_reward(observation: List, is_terminal=False) -> float:
    """
    Reward if note is same key (first note).
    Reward heavily if end on tonic note.
    """
    def is_in_key(ob, key):
        if key is not None and abs(ob - key) % OCTAVE_STEPS in KEY_STEPS:
            return True
        return False

    if len(observation) < 2:
        return NEUTRAL

    key = observation[0] if observation[0] not in {0, 1} else None
    current_note = observation[-1]
    if is_in_key(current_note, key):
        if is_terminal:
            return END_ON_TONIC_REWARD
        return KEY_REWARD

    return NEUTRAL


def get_diversity_reward(observation: List) -> float:
    """
    Reward diversity of notes.
    """
    # cannot calculate correlation if all items the same
    if len(set(observation)) <= 1:
        return NEUTRAL

    # calculate auto-correlation for 1, 2, 3 lags
    # acorr = sm.tsa.acf(observation, nlags=3)
    # first position is 1.0 because lag=0 is always correlated with itself
    # acorr_lag1, acorr_lag2, acorr_lag3 = acorr[1:]
    # diversity_multiplier = len([x for x in acorr if abs(x) < 0.15])

    # try entropy instead
    _, counts = np.unique(observation, return_counts=True)
    h = entropy(counts) / len(observation)
    # max entropy for len observations, counts = 1 for all positions
    max_entropy = entropy([1] * len(observation)) / len(observation)
    # min entropy for len observations
    min_entropy = entropy([1] + [len(observation) - 1]) / len(observation)
    # if h > 0.5 * max_entropy:
    #     return DIVERSITY_REWARD
    if h < 1.25 * min_entropy:
        return DIVERSITY_PENALTY

    return NEUTRAL
