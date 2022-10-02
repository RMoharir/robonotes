"""
This file contains utils for converting to and from MIDI files
"""

from midiutil import MIDIFile
from typing import List


NUM_TRACKS = 1
TRACK = CHANNEL = 0  # single track / channel for now
VOLUME = 100  # fixed volume for now, 0-127
TEMPO = 160  # BPM
DURATION = 1  # 1/16 note, in beats


def note2midi(note: int) -> int:
    """
    Mapping of encoded `note` used in RoboNotesEnv to MIDI pitch.
    Two special encodings used for `note_off` and `no_event`.

    {
        0: 0,  "note_off",
        1: 1,  "no_event",
        2: 48,  C3
        3: 49,  C#3
        4: 50,  D3
        5: 51,  D#3
        ...
        37: 83  # B5
    }
    """
    if note == 0 or note == 1:
        return note
    return note + 46  # offset


def convert_observation_to_midi_sequence(observation: List):
    """
    Convert to MIDI pitch sequence.
    Handle "note_off" and "no_event".

    :return:
    """
    midi_seq = []
    duration = 1
    for curr_ts, curr_notes in enumerate(observation):
        pitch = note2midi(curr_notes)

        if pitch == 0:
            # note_off or rest
            midi_seq.append((pitch, duration))
        elif pitch == 1:
            # add duration to previous note
            if midi_seq:
                pitch, duration = midi_seq[-1]
                midi_seq[-1] = (pitch, duration + 1)
            else:
                # if first note, treat like a rest
                midi_seq.append((0, duration))
        else:
            assert 48 <= pitch <= 83, f"encountered unexpected pitch: {pitch}"
            midi_seq.append((pitch, duration))
    return midi_seq


def convert_to_midi_file(midi_sequence: List, filepath: str):
    """
    Convert to MIDI file and save at filepath.
    Can use free web tools to play, ex. https://midiplayer.ehubsoft.net/
    :param midi_sequence:
    :param filepath
    :return:
    """
    midifile = MIDIFile(NUM_TRACKS)

    time = 0
    midifile.addTempo(TRACK, time, TEMPO)
    for pitch, duration in midi_sequence:
        if pitch != 0:
            assert 48 <= pitch <= 83, f"encountered unexpected pitch: {pitch}"
            midifile.addNote(TRACK, CHANNEL, pitch, time, duration, VOLUME)

        # if note_off, just increase time
        time += duration

    with open(filepath, "wb") as outfile:
        midifile.writeFile(outfile)
