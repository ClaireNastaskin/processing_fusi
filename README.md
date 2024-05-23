# Anise
functional analysis of fUSI data

## Installation

### Non-Python prerequisites

* Mangrove is currently only supported on macOS Apple Silicon.
* Install [Homebrew](https://brew.sh/) to manage system packages.

### Virtual environment

Create and activate a separate environment for this installation using [conda or mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install). 

```console
mamba create -n Anise python=3.9
```

Activate it by running:

```console
mamba activate Anise
```

Note: when [installing conda](https://www.anaconda.com/download), download the version compatible with your CPU architecture (e.g., Intel vs Apple Silicon).

### Package and dependencies

After activating your Python environment, you can install `Anise` and its dependencies into your environment. From the Anise folder, run:

```console
pip install -e .
```
This install all the dependencies in the pyproject.toml. In addition, you'll also need to install you local mangrove installation:

```console
pip install /path/to/your/mangrove
```
e.g.
```console
pip install /Users/taflalo/code/Mangrove
```

Finally, install the Anise code as a module 
```console
python setup.py install
```