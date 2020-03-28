# stat-utils
A library for various statistical learning and inference methods.

# Installation

## Optional: Create and source virtual environment

Create a virtual environment for this library. Note that Python3 is the expected Python version.
Replace `<path to virtual env>` as needed, e.g., `~/envs/stat_utils`:

`virtualenv --system-site-packages <path to virtual env>`

Activate the virtual environment:

`source <path to virtual env>/bin/activate`

## Install package
We use `-e` for editable mode (you may remove this option if you prefer):

`pip install -e .`

# Test
All unit tests should pass. Run unit tests through the following command:

`pytest`
