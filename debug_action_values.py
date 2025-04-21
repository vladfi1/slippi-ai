import os
import tempfile
import zipfile
import sys
import requests
import peppi_py
import melee
import numpy as np
from collections import defaultdict

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

def get_peppi_actions(slp_file):
    print(f"Getting peppi actions from {slp_file}")
    
    peppi_game = peppi_py.read_slippi(slp_file)
    
    actions = []
    
    try:
        if hasattr(peppi_game.frames, 'ports') and len(peppi_game.frames.ports) > 0:
            port_data = peppi_game.frames.ports[0]
            
            if hasattr(port_data, 'leader') and hasattr(port_data.leader, 'post'):
                post = port_data.leader.post
                
                if hasattr(post, 'state'):
                    state = post.state
                    
                    if hasattr(state, 'to_numpy'):
                        actions = state.to_numpy()
                    elif isinstance(state, (list, tuple)):
                        actions = np.array(state)
                    else:
                        actions = np.array([state])
    except Exception as e:
        print(f"Error extracting peppi actions: {e}")
    
    return actions

def get_libmelee_actions(slp_file):
    print(f"Getting libmelee actions from {slp_file}")
    
    console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
    console.connect()
    
    actions = []
    
    while True:
        gamestate = console.step()
        if gamestate is None:
            break
        
        if gamestate.players and 0 in gamestate.players:
            actions.append(gamestate.players[0].action.value)
    
    return np.array(actions)

def analyze_action_mapping(slp_file):
    print(f"Analyzing action mapping in {slp_file}")
    
    peppi_actions = get_peppi_actions(slp_file)
    libmelee_actions = get_libmelee_actions(slp_file)
    
    print(f"Peppi actions shape: {peppi_actions.shape}")
    print(f"Libmelee actions shape: {libmelee_actions.shape}")
    
    min_len = min(len(peppi_actions), len(libmelee_actions))
    
    if min_len == 0:
        print("No actions to compare")
        return None
    
    diffs = []
    for i in range(min_len):
        peppi_val = peppi_actions[i]
        libmelee_val = libmelee_actions[i]
        diff = libmelee_val - peppi_val  # Note: we're calculating libmelee - peppi
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
        for diff, count in sorted(diff_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"Offset {diff}: {count} occurrences ({percentage:.2f}%)")
        
        most_common_diff = max(diff_counts, key=diff_counts.get)
        most_common_percentage = (diff_counts[most_common_diff] / total) * 100
        
        if most_common_percentage > 50:
            print(f"DOMINANT OFFSET FOUND: {most_common_diff} ({most_common_percentage:.2f}%)")
            return most_common_diff
        
        print("\nTrying to find a mapping between peppi and libmelee action values...")
        
        mapping = defaultdict(list)
        for i in range(min_len):
            peppi_val = peppi_actions[i]
            libmelee_val = libmelee_actions[i]
            mapping[peppi_val].append(libmelee_val)
        
        for peppi_val, libmelee_vals in sorted(mapping.items())[:20]:
            val_counts = {}
            for val in libmelee_vals:
                if val in val_counts:
                    val_counts[val] += 1
                else:
                    val_counts[val] = 1
            
            most_common_val = max(val_counts, key=val_counts.get)
            percentage = (val_counts[most_common_val] / len(libmelee_vals)) * 100
            
            print(f"Peppi {peppi_val} -> Libmelee {most_common_val} ({percentage:.2f}%)")
            
            try:
                peppi_action = melee.Action(peppi_val)
                libmelee_action = melee.Action(most_common_val)
                print(f"  {peppi_action.name} -> {libmelee_action.name}")
            except ValueError:
                pass
        
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
