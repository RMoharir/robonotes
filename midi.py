"""
This file contains utils for converting to and from MIDI files
"""

from midiutil import MIDIFile
from typing import List


NUM_TRACKS = 1
TRACK = CHANNEL = 0  # single track / channel for now
VOLUME = 100  # fixed volume for now, 0-127
TEMPO = 60  # BPM
DURATION = 1  # 1/16 note, in beats

NOTE2MIDI = {
    0: "note_off",
    1: "no_event",
    2: 48,  # C3
    # ...
    37: 83  # B5
}


def convert_observation_to_midi_sequence(observation: List[List[int]]):
    """
    Convert to MIDI pitch sequence.
    Handle "note_off" and "no_event".

    :return:
    """
    midi_seq = []
    duration = 1
    held_note = None
    for curr_ts, curr_notes in enumerate(observation):
        if len(curr_notes) > 1:
            raise ValueError("Multiple tracks not supported yet. Expect only one note per ts.")
        midi_note = NOTE2MIDI[curr_notes[0]]
        next_midi_note = observation[curr_ts + 1]
        if next_midi_note == "no_event":
            # special case: we dont add curr_note just yet since we are extending the duration
            duration += 1
            held_note = midi_note
            continue

        if held_note:
            # note is held from previous ts, and now released
            midi_seq.append((held_note, duration))
            # reset duration, held_note
            held_note = None
            duration = 1

        # add current ts note, either "note_off" (rest) or a midi pitch
        midi_seq.append((midi_note, duration))

    return midi_seq


def convert_to_midi_file(midi_sequence: List, filepath: str):
    """
    An encoding
    :param midi_sequence:
    :return:
    """
    midifile = MIDIFile(NUM_TRACKS)

    time = 0
    midifile.addTempo(TRACK, time, TEMPO)
    for pitch, duration in midi_sequence:
        if pitch != "note_off":
            midifile.addNote(TRACK, CHANNEL, pitch, time, duration, VOLUME)

        # if note_off, just increase time
        time += duration

    with open(filepath, "wb") as outfile:
        midifile.writeFile(outfile)
