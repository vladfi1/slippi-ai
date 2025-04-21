import os
import tempfile
import zipfile
import sys
import requests
import peppi_py
import melee
import numpy as np

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

def compare_action_states(slp_file):
    print(f"Comparing action states in {slp_file}")
    
    peppi_game = peppi_py.read_slippi(slp_file)
    
    console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
    console.connect()
    libmelee_gamestate = console.step()
    
    peppi_frame = peppi_game.frames
    
    peppi_ports = sorted(p.port for p in peppi_game.start.players)
    libmelee_ports = sorted(libmelee_gamestate.players)
    
    print(f"Peppi ports: {peppi_ports}")
    print(f"Libmelee ports: {libmelee_ports}")
    
    for port in peppi_ports:
        port_str = str(port)
        if hasattr(peppi_frame.ports, port_str):
            peppi_port_data = getattr(peppi_frame.ports, port_str)
            peppi_action = peppi_port_data.leader.post.state
            
            try:
                melee_action = melee.Action(peppi_action).value
                print(f"Port {port}: Peppi action {peppi_action} -> Melee action {melee_action}")
            except ValueError:
                print(f"Port {port}: Peppi action {peppi_action} is not a valid melee.Action")
            
            if int(port_str[1:]) in libmelee_gamestate.players:
                libmelee_action = libmelee_gamestate.players[int(port_str[1:])].action.value
                print(f"Port {port}: Libmelee action {libmelee_action}")
                
                if peppi_action != libmelee_action:
                    print(f"  MISMATCH: Peppi {peppi_action} != Libmelee {libmelee_action}")
            else:
                print(f"Port {port} not found in libmelee gamestate")

def explore_action_mapping():
    """Explore the mapping between peppi action states and melee.Action enum values."""
    print("Exploring action mapping...")
    
    melee_actions = {action.value: action.name for action in melee.Action}
    print(f"Total melee.Action enum values: {len(melee_actions)}")
    
    print("Example melee.Action values:")
    for i, (value, name) in enumerate(sorted(melee_actions.items())[:10]):
        print(f"  {value}: {name}")
    
    print("\nChecking for patterns or offsets...")
    
    def map_action(peppi_action):
        """Try to map peppi action to melee.Action."""
        try:
            return melee.Action(peppi_action).value
        except ValueError:
            for offset in [-1, 1, -2, 2]:
                try:
                    return melee.Action(peppi_action + offset).value
                except ValueError:
                    pass
            return melee.Action.STANDING.value

    print("Example mappings:")
    for test_value in range(1, 100, 10):
        try:
            mapped = map_action(test_value)
            print(f"  Peppi {test_value} -> Melee {mapped} ({melee.Action(mapped).name})")
        except Exception as e:
            print(f"  Error mapping {test_value}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        compare_action_states(slp_file)
        explore_action_mapping()
    else:
        print("No .slp file found")
