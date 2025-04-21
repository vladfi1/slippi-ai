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

def direct_compare(slp_file):
    print(f"Directly comparing action values in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    
    peppi_action = None
    try:
        if hasattr(peppi_game_raw.frames, 'ports') and len(peppi_game_raw.frames.ports) > 0:
            port_data = peppi_game_raw.frames.ports[0]
            if hasattr(port_data, 'leader') and hasattr(port_data.leader, 'post'):
                post = port_data.leader.post
                if hasattr(post, 'state'):
                    peppi_action = post.state
    except Exception as e:
        print(f"Error accessing peppi action: {e}")
    
    print(f"Peppi raw action: {peppi_action}")
    print(f"Peppi action type: {type(peppi_action)}")
    
    console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
    console.connect()
    gamestate = console.step()
    
    libmelee_action = None
    if gamestate and gamestate.players and 0 in gamestate.players:
        libmelee_action = gamestate.players[0].action
    
    print(f"Libmelee action: {libmelee_action}")
    print(f"Libmelee action type: {type(libmelee_action)}")
    
    if peppi_action is not None and libmelee_action is not None:
        for offset in range(-10, 11):
            try:
                if isinstance(peppi_action, (list, tuple)) and len(peppi_action) > 0:
                    adjusted = peppi_action[0] + offset
                else:
                    adjusted = peppi_action + offset
                
                print(f"Offset {offset}: {peppi_action} + {offset} = {adjusted}")
                
                if adjusted == libmelee_action.value:
                    print(f"MATCH FOUND with offset {offset}!")
                    return offset
            except (TypeError, ValueError) as e:
                print(f"Error with offset {offset}: {e}")
    
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        offset = direct_compare(slp_file)
        
        if offset is not None:
            print(f"\nRecommended fix: Add {offset} to peppi action values")
            print("Example code for get_player function:")
            print(f"action_value = np.array([int(state) + {offset}] * num_frames, dtype=np.uint16)")
    else:
        print("No .slp file found")
