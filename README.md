# Imitation learning from Slippi replays

## Minimal installation and setup

This repo requires Tensorflow 2.0 or higher which requires CUDA 10.0 or higher.

An easy way to get started is with Google Cloud Platform. Launch a VM with the 'Deep Learning VM' Debian 9 instance template by Google with CUDA 11.0 and Tensorflow 2.3. It comes with python 3.7.8 and a bunch of standard packages installed. Then, install the rest of the required packages with:

```bash
pip install --user -r requirements.txt
```

A dataset of processed and compressed slippi replays is available at https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B. It is a tarball of zipped pickled slp files. Use

```bash
gdown https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B data/
tar -xf data/AllCompressed.tar
```

to expand it. The folder `data/AllCompressed/` will now contain many zipped pickled and formatted slippi replay files. Another useful dataset is https://drive.google.com/uc?id=1ZIfDgkdQdu-ldCx_34e-VxYJwQCpV-i3 which only contains fox dittos.

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
python train.py
```

### Processing a preexisting dataset of raw slippi replay files

An preexisting dataset of raw slippi replays is available at https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f (27G, unzips to 200G). You can place this in the `data/` folder using `gdown <drive link> <destination>`.

The code relies on a small (~3 MB) sql database which is 'melee_public_slp_dataset.sqlite3' in the `data/` folder.

For updates on this raw slippi replay dataset, the sql database, or the dataset of processed and compressed slippi replays, check the ai channel of the Slippi discord.



