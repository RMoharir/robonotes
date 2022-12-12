"""
This file contains utils for converting to and from MIDI files
"""
import numpy as np
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
        0: 0,  "no_event",
        1: 1,  "note_ff",
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
    def _convert_track(track: List):
        midi_seq = []
        duration = 1
        for curr_ts, curr_notes in enumerate(track):
            pitch = note2midi(curr_notes)

            if pitch == 1:
                # note_off or rest
                midi_seq.append((pitch, duration))
            elif pitch == 0:
                # add duration to previous note
                if midi_seq:
                    prev_pitch, prev_duration = midi_seq[-1]
                    midi_seq[-1] = (prev_pitch, prev_duration + 1)
                else:
                    # if first note, treat like a rest
                    midi_seq.append((1, pitch))
            else:
                assert 48 <= pitch <= 83, f"encountered unexpected pitch: {pitch}"
                midi_seq.append((pitch, duration))
        return midi_seq

    midi_sequences = []
    if type(observation[0]) == list:
        tracks = []
        num_pitches = len(observation[0])

        for track_idx in range(num_pitches):
            tracks.append([obs[track_idx] for obs in observation])

        for track in tracks:
            midi_sequences.append(_convert_track(track))
    else:
        midi_sequences.append(_convert_track(observation))

    return midi_sequences


def convert_to_midi_file(midi_sequences: List[List], filepath: str):
    """
    Convert to MIDI file and save at filepath.
    Can use free web tools to play, ex. https://midiplayer.ehubsoft.net/
    :param midi_sequences:
    :param filepath
    :return:
    """
    midifile = MIDIFile(NUM_TRACKS)
    midifile.addTempo(TRACK, 0, TEMPO)

    def _add_midi_seq_to_midifile(seq: List):
        time = 0
        for pitch, duration in seq:
            if pitch != 1:
                assert 48 <= pitch <= 83, f"encountered unexpected pitch: {pitch}"
                midifile.addNote(TRACK, CHANNEL, pitch, time, duration, VOLUME)

            # if note_off, just increase time
            time += duration

    for midi_sequence in midi_sequences:
        _add_midi_seq_to_midifile(midi_sequence)

    with open(filepath, "wb") as outfile:
        midifile.writeFile(outfile)
