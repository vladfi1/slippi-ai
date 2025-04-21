import os
import tempfile
import zipfile
import sys
import requests
import peppi_py

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

def explore_game_structure(slp_file):
    print(f"Exploring structure of {slp_file}")
    game = peppi_py.read_slippi(slp_file)
    
    print("\nGame object attributes:")
    print(dir(game))
    
    print("\nFrames object type and attributes:")
    print(f"Type of game.frames: {type(game.frames)}")
    print(f"Dir of game.frames: {dir(game.frames)}")
    
    print("\nExploring how to access multiple frames:")
    
    print("\nChecking if game object has methods to get all frames:")
    for attr in dir(game):
        if 'frame' in attr.lower() and callable(getattr(game, attr)):
            print(f"Found potential method: {attr}")
    
    print("\nChecking if game object has attributes that might contain all frames:")
    for attr in dir(game):
        if 'frame' in attr.lower() and not callable(getattr(game, attr)):
            print(f"Found potential attribute: {attr}")
    
    print("\nTrying to access frames by index:")
    try:
        frame_0 = game.frames
        print(f"Frame 0 ID: {frame_0.id}")
        
        print("Trying to access next frame...")
        if hasattr(frame_0, 'next_frame'):
            frame_1 = frame_0.next_frame
            print(f"Frame 1 ID: {frame_1.id}")
        else:
            print("No next_frame method/attribute found")
    except Exception as e:
        print(f"Error accessing frames by index: {e}")
    
    print("\nTrying to understand the structure of the game object:")
    try:
        print(f"Game has {len(game.start.players)} players")
        print(f"First player port: {game.start.players[0].port}")
        
        print("\nAccessing frame data through game object:")
        frame = game.frames
        print(f"Frame ID: {frame.id}")
        
        print("\nAccessing ports in the frame:")
        port_name = game.start.players[0].port
        if hasattr(frame.ports, port_name):
            port_data = getattr(frame.ports, port_name)
            print(f"Port {port_name} exists in frame {frame.id}")
            print(f"Port data attributes: {dir(port_data)}")
            
            if hasattr(port_data, 'leader'):
                leader = port_data.leader
                print(f"Leader attributes: {dir(leader)}")
                
                if hasattr(leader, 'post'):
                    post = leader.post
                    print(f"Post attributes: {dir(post)}")
                    
                    if hasattr(post, 'character'):
                        print(f"Character: {post.character}")
        else:
            print(f"Port {port_name} does not exist in frame {frame.id}")
    except Exception as e:
        print(f"Error exploring game structure: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slp_file = sys.argv[1]
    else:
        slp_file = download_test_dataset()
    
    if slp_file:
        explore_game_structure(slp_file)
    else:
        print("No .slp file found")
