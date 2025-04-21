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
import tree

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

def compare_action_values(slp_file):
    print(f"Comparing action values in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    peppi_game = parse_peppi.from_peppi(peppi_game_raw)
    peppi_game_nest = types.array_to_nest(peppi_game)
    
    libmelee_game = parse_libmelee.get_slp(slp_file)
    libmelee_game_nest = types.array_to_nest(libmelee_game)
    
    print("\nComparing action values:")
    
    peppi_shape = peppi_game_nest.p0.action.shape
    libmelee_shape = libmelee_game_nest.p0.action.shape
    print(f"Peppi action shape: {peppi_shape}")
    print(f"Libmelee action shape: {libmelee_shape}")
    
    peppi_type = peppi_game_nest.p0.action.dtype
    libmelee_type = libmelee_game_nest.p0.action.dtype
    print(f"Peppi action dtype: {peppi_type}")
    print(f"Libmelee action dtype: {libmelee_type}")
    
    print("\nFirst 10 action values:")
    for i in range(min(10, len(peppi_game_nest.p0.action))):
        peppi_val = peppi_game_nest.p0.action[i]
        libmelee_val = libmelee_game_nest.p0.action[i]
        diff = peppi_val - libmelee_val
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
    
    diffs = []
    for i in range(min(100, len(peppi_game_nest.p0.action))):
        peppi_val = peppi_game_nest.p0.action[i]
        libmelee_val = libmelee_game_nest.p0.action[i]
        diff = peppi_val - libmelee_val
        diffs.append(diff)
    
    unique_diffs = np.unique(diffs)
    print(f"\nUnique differences: {unique_diffs}")
    
    if len(unique_diffs) == 1:
        print(f"CONSISTENT OFFSET FOUND: {unique_diffs[0]}")
        
        print("\nTrying to apply the offset:")
        offset = unique_diffs[0]
        for i in range(min(5, len(peppi_game_nest.p0.action))):
            peppi_val = peppi_game_nest.p0.action[i]
            adjusted_val = peppi_val - offset
            libmelee_val = libmelee_game_nest.p0.action[i]
            print(f"Frame {i}: Original={peppi_val}, Adjusted={adjusted_val}, Target={libmelee_val}")
            
            try:
                adjusted_action = melee.Action(adjusted_val)
                print(f"  Adjusted action name: {adjusted_action.name}")
            except ValueError:
                print(f"  Adjusted value {adjusted_val} is not a valid melee.Action")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        compare_action_values(slp_file)
    else:
        print("No .slp file found")
