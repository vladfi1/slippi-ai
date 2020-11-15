# Imitation learning from Slippi replays

### Installation and setup

This repo requires Tensorflow 2.0 or higher which requires CUDA 10.0 or higher.

To get started with Google Cloud Platform, you can use the 'Deep Learning VM' instance template by Google with CUDA 11.0 and Tensorflow 2.3. It comes with python 3.7.8 and a bunch of standard packages installed. Then, do:

`pip install --user -r requirements.txt`

The code expects 'melee_public_slp_dataset.sqlite3' in the `/data/` folder, which can be obtained here: https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f/view?usp=sharing (27G, unzips to 200G). Check the ai channel of the Slippi discord for potential updates. 