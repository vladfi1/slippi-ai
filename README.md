# Imitation learning from Slippi replays

## Minimal installation and setup

This repo requires Tensorflow 2.0 or higher which requires CUDA 10.0 or higher.

An easy way to get started is with Google Cloud Platform. Launch a VM with the 'Deep Learning VM' Debian 9 instance template by Google with CUDA 11.0 and Tensorflow 2.3. It comes with python 3.7.8 and a bunch of standard packages installed. Then, install the rest of the required packages with:

```bash
pip install --user -r requirements.txt
```

We use Sacred and MongoDB for logging experiments. While Sacred is installed through requirements.txt, MongoDB needs to be installed separately. Instructions for installing MongoDB on Debian 9 are available here: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-debian/. These commands worked for us:

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
echo "deb http://repo.mongodb.org/apt/debian stretch/mongodb-org/4.4 main" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

To ensure that MongoDB can start at instance startup, run:

```bash
sudo systemctl enable mongod.service
```

To try training a simple model, run

```bash
python scripts/train.py
```

To view the results, you can use [omniboard](https://github.com/vivekratnavel/omniboard), although several [other options](https://github.com/IDSIA/sacred#frontends) are available.

## Training Data

### Old Format

A dataset of processed and compressed slippi replays is available at https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B. It is a tarball of zipped pickled slp files. Use

```bash
gdown 'https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B' -O data/
tar -xf data/AllCompressed.tar
```

to expand it. The folder `data/AllCompressed/` will now contain many zipped pickled and formatted slippi replay files. Another useful dataset is https://drive.google.com/uc?id=1ZIfDgkdQdu-ldCx_34e-VxYJwQCpV-i3 which only contains fox dittos.

### Processing a preexisting dataset of raw slippi replay files
An preexisting dataset of raw slippi replays is available at https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f (27G, unzips to 200G). You can place this in the `data/` folder using `gdown <drive link> <destination>`.

The code relies on a small (~3 MB) sql database which is 'melee_public_slp_dataset.sqlite3' in the `data/` folder.

For updates on this raw slippi replay dataset, the sql database, or the dataset of processed and compressed slippi replays, check the ai channel of the Slippi discord.

### New Format

The old data format had a few issues:
- It was potentially insecure due to the use of pickle.
- It used nests of numpy arrays, lacking any structure or specification.
- Being based on pickle, it was tied to the python language.

The new data format is based on the language-agnostic [Arrow](https://arrow.apache.org/) library and serialization via [Parquet](https://parquet.apache.org/). You can download the new dataset [here](https://slp-replays.s3.amazonaws.com/prod/datasets/pq/games.tar) as a tar archive. It contains 195536 files filtered to be valid singles replays; see `slippi_db.preprocessing.is_training_replay` for what that means. An associated metadata file, also in parquet format, is available [here](https://slp-replays.s3.amazonaws.com/prod/datasets/pq/meta.pq). The metadata file can be loaded as a pandas DataFrame:

```python
import pandas as pd
df = pd.read_parquet('meta.pq')
print(df.columns)
```

To access the game files, you can unzip the tar, or mount it directly using [ratarmount](https://github.com/mxmlnkn/ratarmount). The tar is a flat directory with filenames equal to the md5 hash of the original .slp replay, corresponding to the "key" column in the metadata. Each file is a gzip-compressed parquet table with a single column called "root".

```python
import pyarrow.parquet as pq
table = pq.read_table(game_path)
game = table['root'].combine_chunks()  # pyarrow.StructArray
game[0].as_py()  # nested python dictionary representing the first frame
```

See `slippi_ai/types.py` for utility functions that can manipulate pyarrow objects and convert them to the usual python nests of numpy arrays that are used in machine learning.

## Configuration

Example command configurations:

```bash
python scripts/train.py with network.name=frame_stack_mlp
```

These are some available options:
```
network.name=
    mlp (default)
    frame_stack_mlp
    lstm
    gru
    copier

controller_head.name=
    independent (default)
    autoregressive

# Discrete vs float stick embeddings
controller_head.{independent/autoregressive}.discrete_axis=
    True
    False

network.frame_stack_mlp
    .num_frames=5 (default, any integer >1)
    .frame_delay=0 (default, any integer >=0)

```

## Testing and evaluation

During training, models are saved in directories such as `experiments/<YYYY>-<MM>-<DD>-<hash>/saved_model/saved_model/`. To test the model on a different dataset, run:

```bash
python test.py with saved_model_path=experiments/<YYYY>-<MM>-<DD>-<hash>/saved_model/saved_model/ dataset.subset=all
```

To write out model inputs, output samples, losses, gamestate info, etc to help investigate model behavior through data analysis:
```bash
python debug_test.py with saved_model_path=experiments/<YYYY>-<MM>-<DD>-<hash>/saved_model/saved_model/ dataset.subset=all debug.table_length=<num. rows in resulting pandas dataframe>
```

To evaluate a trained model at playing SSBM in Dolphin, run:

```bash
python eval.py with saved_model_path=experiments/<YYYY>-<MM>-<DD>-<hash>/saved_model/saved_model/ dolphin_path=<path/to/dolphin/> iso_path=<path/to/iso>
```

While Google Cloud Platform VM instances seem to support remote desktop environments with graphical applications (https://cloud.google.com/solutions/chrome-desktop-remote-on-compute-engine), at this time we recommend running dolphin for evaluation with a local computer. We and others have been able to install model evaluation on Linux, MacOS, and Windows.
