# Imitation learning from Slippi replays

## Minimal installation and setup

This repo requires Tensorflow 2.0 or higher which requires CUDA 10.0 or higher.

An easy way to get started is with Google Cloud Platform. Launch a VM with the 'Deep Learning VM' Debian 9 instance template by Google with CUDA 11.0 and Tensorflow 2.3. It comes with python 3.7.8 and a bunch of standard packages installed. Then, install the rest of the required packages with:

```bash
conda env create -f environment.yaml # (optional)
pip install --user -r requirements.txt
```

## Wandb

**Sacred is no longer supported, wandb is the new logging platform used.**

Create an account on [wandb](https://wandb.ai/site), running online for the first time will allow you to login to your account by providing the API key.

Try training a simple model, run

```bash
python scripts/train.py --wandb.mode=online # default is "disabled"
```

Afterwards you can access the data from your teams organization page.

### Additional Options

* **disabled**: **Default**, doesn't log any data to disk or cloud.
* **offline**: Logs data to disk, able to be uploaded to wandb.
* **online**: Logs data live to wandb.

## Training Data

The old data format had a few issues:
- It was potentially insecure due to the use of pickle.
- It used nests of numpy arrays, lacking any structure or specification.
- Being based on pickle, it was tied to the python language.

The new data format is based on the language-agnostic [Arrow](https://arrow.apache.org/) library and serialization via [Parquet](https://parquet.apache.org/). You can download the new dataset [here](https://slp-replays.s3.amazonaws.com/prod/datasets/pq/games.tar) as a tar archive, or use a smaller [test dataset](https://slp-replays.s3.amazonaws.com/prod/datasets/pq/games.tar). The full dataset contains 195536 files filtered to be valid singles replays; see `slippi_db.preprocessing.is_training_replay` for what that means. An associated metadata file, also in parquet format, is available [here](https://slp-replays.s3.amazonaws.com/prod/datasets/pq/meta.pq). The metadata file can be loaded as a pandas DataFrame:

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
python scripts/train.py with dataset.data_dir=path/to/untarred/dir
```

These are some available options:
```
network.name=
    mlp (default)
    frame_stack_mlp
    lstm
    gru

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
