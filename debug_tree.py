import os
import tempfile
import zipfile
import sys
import requests
import peppi_py
import melee
import numpy as np
import tree
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

def debug_tree_structure(slp_file):
    print(f"Debugging tree structure in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    peppi_game = parse_peppi.from_peppi(peppi_game_raw)
    peppi_game_nest = types.array_to_nest(peppi_game)
    
    libmelee_game = parse_libmelee.get_slp(slp_file)
    libmelee_game_nest = types.array_to_nest(libmelee_game)
    
    print("\nPeppi game structure:")
    print_structure(peppi_game_nest)
    
    print("\nLibmelee game structure:")
    print_structure(libmelee_game_nest)
    
    print("\nComparing shapes:")
    compare_shapes(peppi_game_nest, libmelee_game_nest)

def print_structure(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{prefix}{key}: {type(value)}")
            if isinstance(value, (dict, list, tuple)) and not isinstance(value, np.ndarray):
                print_structure(value, prefix + "  ")
            elif isinstance(value, np.ndarray):
                print(f"{prefix}  Shape: {value.shape}, Dtype: {value.dtype}")
    elif isinstance(obj, (list, tuple)) and not isinstance(obj, np.ndarray):
        for i, item in enumerate(obj):
            print(f"{prefix}[{i}]: {type(item)}")
            if isinstance(item, (dict, list, tuple)) and not isinstance(item, np.ndarray):
                print_structure(item, prefix + "  ")
            elif isinstance(item, np.ndarray):
                print(f"{prefix}  Shape: {item.shape}, Dtype: {item.dtype}")

def compare_shapes(peppi, libmelee, path=""):
    if isinstance(peppi, dict) and isinstance(libmelee, dict):
        for key in peppi:
            if key in libmelee:
                new_path = f"{path}.{key}" if path else key
                compare_shapes(peppi[key], libmelee[key], new_path)
            else:
                print(f"Key {key} in peppi but not in libmelee at {path}")
        for key in libmelee:
            if key not in peppi:
                print(f"Key {key} in libmelee but not in peppi at {path}")
    elif isinstance(peppi, np.ndarray) and isinstance(libmelee, np.ndarray):
        if peppi.shape != libmelee.shape:
            print(f"Shape mismatch at {path}: peppi {peppi.shape} vs libmelee {libmelee.shape}")
        if peppi.dtype != libmelee.dtype:
            print(f"Dtype mismatch at {path}: peppi {peppi.dtype} vs libmelee {libmelee.dtype}")
    elif type(peppi) != type(libmelee):
        print(f"Type mismatch at {path}: peppi {type(peppi)} vs libmelee {type(libmelee)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        debug_tree_structure(slp_file)
    else:
        print("No .slp file found")
