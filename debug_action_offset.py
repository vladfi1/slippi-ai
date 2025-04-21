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

def compare_raw_action_values(slp_file):
    print(f"Comparing raw action values in {slp_file}")
    
    peppi_game_raw = peppi_py.read_slippi(slp_file)
    
    print(f"Peppi game type: {type(peppi_game_raw)}")
    print(f"Peppi game dir: {dir(peppi_game_raw)}")
    print(f"Peppi frames type: {type(peppi_game_raw.frames)}")
    print(f"Peppi frames dir: {dir(peppi_game_raw.frames)}")
    
    if hasattr(peppi_game_raw.frames, 'ports'):
        print(f"Peppi ports type: {type(peppi_game_raw.frames.ports)}")
        print(f"Peppi ports dir: {dir(peppi_game_raw.frames.ports)}")
        
        port_data = None
        if isinstance(peppi_game_raw.frames.ports, (list, tuple)) and len(peppi_game_raw.frames.ports) > 0:
            port_data = peppi_game_raw.frames.ports[0]
            print(f"Using first port data from tuple")
            
            if hasattr(port_data, 'leader'):
                leader = port_data.leader
                print(f"Leader type: {type(leader)}")
                print(f"Leader dir: {dir(leader)}")
                
                if hasattr(leader, 'pre'):
                    pre = leader.pre
                    print(f"Pre type: {type(pre)}")
                    print(f"Pre dir: {dir(pre)}")
                
                if hasattr(leader, 'post'):
                    post = leader.post
                    print(f"Post type: {type(post)}")
                    print(f"Post dir: {dir(post)}")
                    
                    if hasattr(post, 'state'):
                        state = post.state
                        print(f"State type: {type(state)}")
                        print(f"State value: {state}")
                        
                        try:
                            action = melee.Action(state)
                            print(f"Converted to melee.Action: {action.name} (value: {action.value})")
                        except ValueError:
                            print(f"Could not convert {state} to melee.Action")
        else:
            print("Ports is not a tuple or is empty")
    
    if port_data:
        print(f"Found port data for port {port_data.port}")
        print(f"Port data type: {type(port_data)}")
        print(f"Port data dir: {dir(port_data)}")
        
        try:
            if hasattr(port_data, 'leader'):
                leader = port_data.leader
                print(f"Leader type: {type(leader)}")
                print(f"Leader dir: {dir(leader)}")
                
                if hasattr(leader, 'post'):
                    post = leader.post
                    print(f"Post type: {type(post)}")
                    print(f"Post dir: {dir(post)}")
                    
                    if hasattr(post, 'state'):
                        peppi_state = post.state
                        print(f"Peppi state type: {type(peppi_state)}")
                        
                        if isinstance(peppi_state, (list, tuple)):
                            print(f"Peppi state is a sequence with {len(peppi_state)} elements")
                            for i, state in enumerate(peppi_state[:5]):
                                print(f"  Element {i}: {state}")
                                try:
                                    action = melee.Action(state)
                                    print(f"    Converted to melee.Action: {action.name} (value: {action.value})")
                                except ValueError:
                                    print(f"    Could not convert {state} to melee.Action")
                        else:
                            print(f"Peppi state is a single value: {peppi_state}")
                            try:
                                action = melee.Action(peppi_state)
                                print(f"  Converted to melee.Action: {action.name} (value: {action.value})")
                            except ValueError:
                                print(f"  Could not convert {peppi_state} to melee.Action")
                    else:
                        print("No 'state' attribute in post")
                else:
                    print("No 'post' attribute in leader")
            else:
                print("No 'leader' attribute in port_data")
        except AttributeError as e:
            print(f"Error accessing peppi state: {e}")
    
    console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
    console.connect()
    libmelee_gamestate = console.step()
    
    if libmelee_gamestate and libmelee_gamestate.players:
        port = next(iter(libmelee_gamestate.players))
        player = libmelee_gamestate.players[port]
        
        print(f"\nLibmelee player for port {port}:")
        print(f"  Action: {player.action}")
        print(f"  Action value: {player.action.value}")
        print(f"  Action name: {player.action.name}")
    
    if port_data and libmelee_gamestate and libmelee_gamestate.players:
        print("\nTrying different offsets:")
        
        peppi_val = peppi_state[0] if isinstance(peppi_state, (list, tuple)) else peppi_state
        libmelee_val = player.action.value
        
        print(f"Peppi value: {peppi_val}, Libmelee value: {libmelee_val}")
        
        for offset in range(-10, 11):
            adjusted = peppi_val + offset
            print(f"Offset {offset}: {peppi_val} + {offset} = {adjusted}")
            
            if adjusted == libmelee_val:
                print(f"MATCH FOUND with offset {offset}!")
            
            try:
                adjusted_action = melee.Action(adjusted)
                print(f"  Valid melee.Action: {adjusted_action.name}")
                
                if adjusted_action.name == player.action.name:
                    print(f"  NAME MATCH with offset {offset}!")
            except ValueError:
                pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        compare_raw_action_values(slp_file)
    else:
        print("No .slp file found")
