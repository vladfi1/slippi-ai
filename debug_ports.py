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

def explore_ports_structure(slp_file):
    print(f"Exploring ports structure in {slp_file}")
    
    peppi_game = peppi_py.read_slippi(slp_file)
    
    peppi_frame = peppi_game.frames
    
    peppi_ports = sorted(p.port for p in peppi_game.start.players)
    
    print(f"Peppi ports: {peppi_ports}")
    
    print("\nExamining peppi_frame.ports structure:")
    print(f"Type: {type(peppi_frame.ports)}")
    
    if isinstance(peppi_frame.ports, tuple):
        print(f"Length: {len(peppi_frame.ports)}")
        
        if len(peppi_frame.ports) > 0:
            first_port = peppi_frame.ports[0]
            print(f"\nFirst port data:")
            print(f"Type: {type(first_port)}")
            print(f"Dir: {dir(first_port)}")
            
            if hasattr(first_port, 'port'):
                print(f"Port: {first_port.port}")
            
            if hasattr(first_port, 'leader'):
                leader = first_port.leader
                print(f"\nLeader data:")
                print(f"Type: {type(leader)}")
                print(f"Dir: {dir(leader)}")
                
                if hasattr(leader, 'post'):
                    post = leader.post
                    print(f"\nPost data:")
                    print(f"Type: {type(post)}")
                    print(f"Dir: {dir(post)}")
                    
                    if hasattr(post, 'state'):
                        print(f"State: {post.state}")
                        
                        try:
                            action = melee.Action(post.state)
                            print(f"Converted to melee.Action: {action.name} (value: {action.value})")
                        except ValueError:
                            print(f"Could not convert {post.state} to melee.Action")
    
    print("\nComparing with libmelee:")
    console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
    console.connect()
    libmelee_gamestate = console.step()
    
    libmelee_ports = sorted(libmelee_gamestate.players)
    print(f"Libmelee ports: {libmelee_ports}")
    
    print("\nComparing action states:")
    for port_idx, port in enumerate(peppi_ports):
        port_int = int(str(port)[1:])
        print(f"\nPort {port} (int: {port_int}):")
        
        peppi_port_data = None
        for p in peppi_frame.ports:
            if hasattr(p, 'port') and p.port == port:
                peppi_port_data = p
                break
        
        if peppi_port_data:
            peppi_action = peppi_port_data.leader.post.state
            print(f"Peppi action: {peppi_action}")
            
            if port_int in libmelee_gamestate.players:
                libmelee_action = libmelee_gamestate.players[port_int].action.value
                print(f"Libmelee action: {libmelee_action}")
                
                if peppi_action != libmelee_action:
                    print(f"MISMATCH: Difference = {peppi_action - libmelee_action}")
                    
                    for offset in [-1, 1, -2, 2]:
                        adjusted = peppi_action + offset
                        print(f"With offset {offset}: {adjusted} vs {libmelee_action}")
                        if adjusted == libmelee_action:
                            print(f"MATCH with offset {offset}!")
                else:
                    print("MATCH!")
            else:
                print(f"Port {port_int} not found in libmelee gamestate")
        else:
            print(f"Port {port} not found in peppi_frame.ports")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        explore_ports_structure(slp_file)
    else:
        print("No .slp file found")
