import mido
from typing import List
from midi_writer import convert_to_midi_file
import numpy as np
import os
import glob
from tqdm import tqdm


def midi2note(note: int) -> int:
    return note - 46  # offset


def load_midi_files(midi_dir):
    print("Loading MIDI files")
    midi_files = [y for x in os.walk(midi_dir) for y in glob.glob(os.path.join(x[0], "*.mid"))]
    print(f"Found {len(midi_files)} midi files")
    seqs = []
    errors = 0
    for filepath in tqdm(midi_files):
        try:
            midfile = mido.MidiFile(filepath, clip=True)
            seqs.append(convert_midi_to_seq(midfile))
        except Exception as e:
            print(e)
            errors += 1
            continue
    print(f"Finished loading MIDI files with {errors} errors.")
    return seqs


def convert_midi_to_seq(midfile) -> List:
    num_tracks = len(midfile.tracks)
    selected_track = None
    num_notes = 0
    for i in range(num_tracks):
        track = midfile.tracks[i]
        track_notes = [t for t in track
                       if t.is_meta is False
                       and t.type == 'note_on'
                       and t.velocity > 0
                       and 48 <= t.note <= 83]
        if len(track_notes) > num_notes:
            selected_track = track_notes
            num_notes = len(track_notes)

    return [midi2note(msg.note) for msg in selected_track]


if __name__ == '__main__':
    # in_filepath = 'test.mid'
    # out_filepath = 'out_test.mid'
    #
    # mid = mido.MidiFile(in_filepath, clip=True)
    # seq = convert_midi_to_seq(mid)
    # convert_to_midi_file([seq], out_filepath)
    mididir = "data/adl-piano-midi"
    midiout = "data/midi_arrays.npy"
    seqs = load_midi_files(mididir)
    np.save(midiout, seqs)
    print(f"total number of seqs: {len(seqs)}")
    print(f"samples: {seqs[-1]}")
