"""Prepare parsing in the local filesystem.

Most people looking to run `parse_local.py` have the slippi files they want to 
use in a collection of directories under one source directory, something like:
.
├── 2024-04
│   ├── Game_20250404T110624.slp
│   ├── Game_20250404T110954.slp
│   └── ... more slp files
├── 2024-05
│   ├── Game_20250404T110624.slp
│   ├── Game_20250404T110954.slp
│   └── ... more slp files
└── ... more directories

This script accepts a path to the root of such a directory, and constructs the
directory which `parse_local.py` expects:
.
├── Parsed
└── Raw
    ├── 2024-04.7z
    ├── 2024-05.7z
    └── ... more 7z archives

The resulting directory can then be passed directly to `parse_local.py` to 
generate the files in `Parsed`, as well as `meta.json` and `parsed.pkl`
"""

import os
import shutil
import subprocess

from absl import app, flags

FLAGS = flags.FLAGS


def seven_zip_exists_in_path():
    path = shutil.which("7z")
    return path is not None

def validate_source_directory(source):
    if not os.path.exists(source):
        raise ValueError(f'Failed to find a directory at {source}')

    if not os.path.isdir(source):
        raise ValueError(f'Found something at {source} but it is not a directory')

def create_destination_directory(dest):
    try:
        os.mkdir(dest)
    except FileExistsError:
        return ValueError(f'Failed to create zip root directory at {dest}, something already exists there.')
    except FileNotFoundError:
        return ValueError(f'Failed to create zip root directory at {dest}, parent directory not found.')

    os.mkdir(os.path.join(dest, 'Parsed'))
    os.mkdir(os.path.join(dest, 'Raw'))

def run_preparation(source, dest):
    if not os.path.isabs(source):
        source = os.path.join(os.getcwd(), source)
    if not os.path.isabs(dest):
        dest = os.path.join(os.getcwd(), dest)

    if not seven_zip_exists_in_path():
        raise Exception('Couldn\'t find 7z in path, install it for your platform')

    validate_source_directory(source)

    create_destination_directory(dest)

    # create a list of all directories found under source
    directories = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]

    if len(directories) == 0:
        print(f'No directories contained in {source}, no work to be done.')
        return

    # for each directory under source, create a 7z archive in the destination
    for d in directories:
        source_dir = os.path.join(source, d)
        destination_archive = os.path.join(dest, 'Raw', f'{d}.7z')
        print(f'PROCESSING: {source_dir} and creating an archive at {destination_archive}')

        command = f'7z a -t7z -mx=5 "{destination_archive}" "{source_dir}"'
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            print(f'WARNING: failed to create 7z archive for: {d}')
            print(process.stderr)

def main(_):
    run_preparation(FLAGS.slp_root, FLAGS.zip_root)

if __name__ == '__main__':
    SLP_ROOT = flags.DEFINE_string('slp_root',
        None,
        'root directory containing slippi files',
        required=True)

    ZIP_ROOT = flags.DEFINE_string('zip_root',
        None,
        'destination root directory where the archives will be placed',
        required=True)

    app.run(main)

