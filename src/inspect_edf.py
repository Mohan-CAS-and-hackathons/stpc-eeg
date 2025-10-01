# src/inspect_edf.py
import mne
import os
import argparse
from collections import Counter

def inspect_edf_file(file_path):
    """
    Reads an .edf file and prints a detailed report about its structure,
    focusing on channel names and montage compatibility.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print("="*80)
    print(f"INSPECTION REPORT FOR: {os.path.basename(file_path)}")
    print("="*80)

    try:
        # Load the file header without loading all the data (fast)
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

        # --- Basic Information ---
        print("\n[ General File Info ]")
        print(f"  - Number of Channels: {len(raw.ch_names)}")
        print(f"  - Sampling Frequency: {raw.info['sfreq']} Hz")
        print(f"  - Duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")

        # --- Channel Name Analysis ---
        raw_ch_names = raw.ch_names
        print("\n[ Channel Name Analysis ]")
        
        # Check for duplicates
        counts = Counter(raw_ch_names)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        if duplicates:
            print(f"  - Found {len(duplicates)} duplicate channel name(s):")
            for name, count in duplicates.items():
                print(f"    - '{name}' appears {count} times")
        else:
            print("  - No duplicate channel names found.")

        # --- Montage Compatibility Analysis ---
        print("\n[ Montage Compatibility (standard_1020) ]")
        montage = mne.channels.make_standard_montage('standard_1020')
        montage_channels_upper = {ch.upper() for ch in montage.ch_names}
        
        monopolar_matches = 0
        bipolar_matches = 0
        
        mapped_monopolar_names = set()
        
        for ch_name in raw_ch_names:
            ch_upper = ch_name.upper()
            
            # Check for direct bipolar match (e.g., 'FP1-F7')
            if ch_upper in montage_channels_upper:
                bipolar_matches += 1
            
            # Check for monopolar match by splitting the name
            # This is how we can map 'FP1-F7' to 'FP1'
            mono_name = ch_upper.split('-')[0]
            if mono_name in montage_channels_upper:
                monopolar_matches += 1
                mapped_monopolar_names.add(mono_name)

        print(f"  - The standard_1020 montage contains {len(montage_channels_upper)} channel locations.")
        print(f"  - Found {len(mapped_monopolar_names)} unique monopolar channels in this file that can be mapped to the montage.")
        print(f"  - This is the information our plotting script needs to work correctly.")

        # --- Full Channel List ---
        print("\n[ Raw Channel List (as found in file) ]")
        for i, name in enumerate(raw_ch_names):
            print(f"  {i+1:02d}: {name}")
        
        print("\n" + "="*80)
        print("Inspection Complete.")

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A diagnostic tool to inspect the contents of a single .edf file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "file_path",
        help="The full path to the .edf file you want to inspect."
    )
    args = parser.parse_args()
    inspect_edf_file(args.file_path)

    