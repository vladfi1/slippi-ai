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
    
    slp_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".slp"):
                slp_files.append(os.path.join(root, file))
    
    return slp_files[:5]  # Return first 5 files for analysis

def get_peppi_actions(slp_file):
    print(f"Getting peppi actions from {slp_file}")
    
    try:
        peppi_game = peppi_py.read_slippi(slp_file)
        
        if hasattr(peppi_game.frames, 'ports') and hasattr(peppi_game.frames.ports, '0'):
            port_data = peppi_game.frames.ports[0]
            
            if hasattr(port_data, 'leader') and hasattr(port_data.leader, 'post'):
                post = port_data.leader.post
                
                if hasattr(post, 'state'):
                    state = post.state
                    
                    if hasattr(state, 'to_numpy'):
                        return state.to_numpy()
                    elif isinstance(state, (list, tuple)):
                        return np.array(state)
                    else:
                        return np.array([state])
    except Exception as e:
        print(f"Error extracting peppi actions: {e}")
    
    return np.array([])

def get_libmelee_actions(slp_file):
    print(f"Getting libmelee actions from {slp_file}")
    
    try:
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
    except Exception as e:
        print(f"Error extracting libmelee actions: {e}")
    
    return np.array([])

def analyze_action_ranges(slp_files):
    print(f"Analyzing action ranges in {len(slp_files)} files")
    
    all_peppi_actions = []
    all_libmelee_actions = []
    
    for slp_file in slp_files:
        peppi_actions = get_peppi_actions(slp_file)
        libmelee_actions = get_libmelee_actions(slp_file)
        
        if len(peppi_actions) > 0 and len(libmelee_actions) > 0:
            min_len = min(len(peppi_actions), len(libmelee_actions))
            all_peppi_actions.extend(peppi_actions[:min_len])
            all_libmelee_actions.extend(libmelee_actions[:min_len])
    
    if not all_peppi_actions or not all_libmelee_actions:
        print("No actions to analyze")
        return
    
    peppi_ranges = defaultdict(list)
    libmelee_ranges = defaultdict(list)
    
    for p_action, l_action in zip(all_peppi_actions, all_libmelee_actions):
        p_range = p_action // 100 * 100
        l_range = l_action // 100 * 100
        
        peppi_ranges[p_range].append(p_action)
        libmelee_ranges[l_range].append(l_action)
    
    print("\nPeppi action ranges:")
    for p_range, actions in sorted(peppi_ranges.items()):
        unique_actions = sorted(set(actions))
        print(f"Range {p_range}-{p_range+99}: {len(unique_actions)} unique actions")
        print(f"  Sample: {unique_actions[:10]}")
    
    print("\nLibmelee action ranges:")
    for l_range, actions in sorted(libmelee_ranges.items()):
        unique_actions = sorted(set(actions))
        print(f"Range {l_range}-{l_range+99}: {len(unique_actions)} unique actions")
        print(f"  Sample: {unique_actions[:10]}")
    
    print("\nAnalyzing range-specific offsets:")
    range_offsets = {}
    
    for p_range in sorted(peppi_ranges.keys()):
        p_actions = peppi_ranges[p_range]
        
        for l_range in sorted(libmelee_ranges.keys()):
            l_actions = libmelee_ranges[l_range]
            
            offsets = []
            for p in p_actions:
                for l in l_actions:
                    offsets.append(l - p)
            
            if offsets:
                offset_counts = {}
                for offset in offsets:
                    if offset in offset_counts:
                        offset_counts[offset] += 1
                    else:
                        offset_counts[offset] = 1
                
                most_common_offset = max(offset_counts, key=offset_counts.get)
                most_common_count = offset_counts[most_common_offset]
                percentage = (most_common_count / len(offsets)) * 100
                
                if percentage > 50:
                    range_offsets[p_range] = (l_range, most_common_offset, percentage)
                    print(f"Peppi range {p_range}-{p_range+99} -> Libmelee range {l_range}-{l_range+99}: offset {most_common_offset} ({percentage:.2f}%)")
    
    return range_offsets

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_files = [sys.argv[1]]
    else:
        slp_files = download_test_dataset()
    
    if slp_files:
        range_offsets = analyze_action_ranges(slp_files)
        
        if range_offsets:
            print("\nRecommended fix: Use range-specific offsets")
            print("Example code for get_player function:")
            print("def get_action_offset(action_value):")
            for p_range, (l_range, offset, _) in sorted(range_offsets.items()):
                print(f"    if {p_range} <= action_value < {p_range+100}:")
                print(f"        return {offset}")
            print("    return 0  # Default offset")
    else:
        print("No .slp files found")
