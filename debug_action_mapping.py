import os
import tempfile
import zipfile
import sys
import requests
import peppi_py
import melee
import numpy as np
from slippi_db import parse_peppi
from slippi_db import parse_libmelee
from slippi_ai import types

def download_test_dataset():
    TEST_DATASET_URL = "https://www.dropbox.com/scl/fi/xbja5vqqlg3m8jutyjcn7/TestDataset-32.zip?rlkey=nha6ycc6npr3wmxzickeyqpfh&st=i87xxfxk&dl=1"
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "test_dataset.zip")
    
    print(f"Downloading from {TEST_DATASET_URL} to {zip_path}...")
    response = requests.get(TEST_DATASET_URL, stream=True)
    response.raise_for_status()
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download complete: {zip_path}")
    
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Extraction complete: {extract_dir}")
    
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".slp"):
                return os.path.join(root, file)
    
    return None

def analyze_action_mapping(slp_file):
    print(f"Analyzing action mapping in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    peppi_game = parse_peppi.from_peppi(peppi_game_raw)
    peppi_game_nest = types.array_to_nest(peppi_game)
    
    libmelee_game = parse_libmelee.get_slp(slp_file)
    libmelee_game_nest = types.array_to_nest(libmelee_game)
    
    peppi_actions = peppi_game_nest.p0.action
    libmelee_actions = libmelee_game_nest.p0.action
    
    print(f"Peppi actions shape: {peppi_actions.shape}")
    print(f"Libmelee actions shape: {libmelee_actions.shape}")
    
    min_len = min(len(peppi_actions), len(libmelee_actions))
    
    diffs = []
    for i in range(min_len):
        peppi_val = peppi_actions[i]
        libmelee_val = libmelee_actions[i]
        diff = peppi_val - libmelee_val
        diffs.append(diff)
        
        if i < 10:  # Print first 10 values
            print(f"Frame {i}: Peppi={peppi_val}, Libmelee={libmelee_val}, Diff={diff}")
            
            try:
                peppi_action = melee.Action(peppi_val)
                print(f"  Peppi action name: {peppi_action.name}")
            except ValueError:
                print(f"  Peppi value {peppi_val} is not a valid melee.Action")
                
            try:
                libmelee_action = melee.Action(libmelee_val)
                print(f"  Libmelee action name: {libmelee_action.name}")
            except ValueError:
                print(f"  Libmelee value {libmelee_val} is not a valid melee.Action")
    
    unique_diffs = np.unique(diffs)
    print(f"\nUnique differences: {unique_diffs}")
    
    if len(unique_diffs) == 1:
        print(f"CONSISTENT OFFSET FOUND: {unique_diffs[0]}")
        return unique_diffs[0]
    else:
        print("No consistent offset found")
        
        diff_counts = {}
        for diff in diffs:
            if diff in diff_counts:
                diff_counts[diff] += 1
            else:
                diff_counts[diff] = 1
        
        total = len(diffs)
        for diff, count in diff_counts.items():
            percentage = (count / total) * 100
            print(f"Offset {diff}: {count} occurrences ({percentage:.2f}%)")
        
        most_common_diff = max(diff_counts, key=diff_counts.get)
        most_common_percentage = (diff_counts[most_common_diff] / total) * 100
        
        if most_common_percentage > 90:
            print(f"DOMINANT OFFSET FOUND: {most_common_diff} ({most_common_percentage:.2f}%)")
            return most_common_diff
        
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        offset = analyze_action_mapping(slp_file)
        
        if offset is not None:
            print(f"\nRecommended fix: Add {offset} to peppi action values")
            print("Example code for get_player function:")
            print(f"action_value = np.array([int(state) + {offset}] * num_frames, dtype=np.uint16)")
    else:
        print("No .slp file found")
