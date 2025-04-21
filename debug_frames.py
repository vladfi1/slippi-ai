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

def compare_frames(slp_file):
    print(f"Comparing frames in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    peppi_game = parse_peppi.from_peppi(peppi_game_raw)
    peppi_game_nest = types.array_to_nest(peppi_game)
    
    libmelee_game = parse_libmelee.get_slp(slp_file)
    libmelee_game_nest = types.array_to_nest(libmelee_game)
    
    print("\nComparing first 5 frames:")
    for i in range(min(5, len(peppi_game_nest.stage))):
        print(f"\nFrame {i}:")
        
        peppi_action = peppi_game_nest.p0.action[i]
        libmelee_action = libmelee_game_nest.p0.action[i]
        
        print(f"  p0 action: Peppi={peppi_action}, Libmelee={libmelee_action}")
        
        if peppi_action != libmelee_action:
            print(f"  MISMATCH: Difference = {peppi_action - libmelee_action}")
            
            if i > 0:
                prev_peppi = peppi_game_nest.p0.action[i-1]
                prev_libmelee = libmelee_game_nest.p0.action[i-1]
                print(f"  Previous frame: Peppi={prev_peppi}, Libmelee={prev_libmelee}, Diff={prev_peppi - prev_libmelee}")
    
    print("\nChecking for consistent offset:")
    diffs = []
    for i in range(min(100, len(peppi_game_nest.stage))):
        peppi_action = peppi_game_nest.p0.action[i]
        libmelee_action = libmelee_game_nest.p0.action[i]
        diff = peppi_action - libmelee_action
        diffs.append(diff)
    
    unique_diffs = np.unique(diffs)
    print(f"Unique differences: {unique_diffs}")
    
    if len(unique_diffs) == 1:
        print(f"CONSISTENT OFFSET FOUND: {unique_diffs[0]}")
    elif len(unique_diffs) <= 5:
        print("Small number of different offsets found. Might be related to action state categories.")
        for diff in unique_diffs:
            count = np.sum(diffs == diff)
            print(f"  Offset {diff}: {count} occurrences ({count/len(diffs)*100:.1f}%)")
    else:
        print("No consistent offset found.")

def explore_frame_structure(slp_file):
    """Explore the structure of frames in peppi-py and libmelee."""
    print(f"\nExploring frame structure in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    
    peppi_frame = peppi_game_raw.frames
    
    print("\nPeppi frame structure:")
    print(f"Type of peppi_frame: {type(peppi_frame)}")
    print(f"Dir of peppi_frame: {dir(peppi_frame)}")
    
    if hasattr(peppi_frame, 'id'):
        print(f"peppi_frame.id type: {type(peppi_frame.id)}")
        print(f"peppi_frame.id length: {len(peppi_frame.id)}")
        print(f"First few frame IDs: {peppi_frame.id[:5]}")
    
    if hasattr(peppi_frame, 'ports'):
        print(f"peppi_frame.ports type: {type(peppi_frame.ports)}")
        print(f"peppi_frame.ports dir: {dir(peppi_frame.ports)}")
        
        port_names = sorted(p.port for p in peppi_game_raw.start.players)
        for port in port_names:
            port_str = str(port)
            if hasattr(peppi_frame.ports, port_str):
                port_data = getattr(peppi_frame.ports, port_str)
                print(f"\nPort {port} data:")
                print(f"  Type: {type(port_data)}")
                print(f"  Dir: {dir(port_data)}")
                
                if hasattr(port_data, 'leader'):
                    leader = port_data.leader
                    print(f"  Leader type: {type(leader)}")
                    print(f"  Leader dir: {dir(leader)}")
                    
                    if hasattr(leader, 'post'):
                        post = leader.post
                        print(f"  Post type: {type(post)}")
                        print(f"  Post dir: {dir(post)}")
                        
                        if hasattr(post, 'state'):
                            print(f"  State type: {type(post.state)}")
                            print(f"  First few states: {[post.state for _ in range(5)]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        explore_frame_structure(slp_file)
        compare_frames(slp_file)
    else:
        print("No .slp file found")
