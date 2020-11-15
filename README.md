# Imitation learning from Slippi replays

### Installation and setup

This repo requires Tensorflow 2.0 or higher which requires CUDA 10.0 or higher.

To get started with Google Cloud Platform, you can use the 'Deep Learning VM' instance template by Google with CUDA 11.0 and Tensorflow 2.3. It comes with python 3.7.8 and a bunch of standard packages installed. Then, do:

`pip install --user -r requirements.txt`

An preexisting dataset of raw slippi replays is available at https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f (27G, unzips to 200G; check the ai channel of the Slippi discord for potential updates). You can place this in the `data/` folder using `gdown <drive link> <destination>`.

The code relies on a small (~3 MB) sql database which is 'melee_public_slp_dataset.sqlite3' in the `data/` folder.

A dataset of processed and compressed slippi replays is available at https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B. It is a tarball of zipped pickled slp files. Use

`gdown https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B data/`
`tar -xf data/AllCompressed.tar`

to expand it, yielding zipped pickled slp files for training ML models. To try training a simple model, run

`python train.py`.

