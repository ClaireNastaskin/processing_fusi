# Anise
functional analysis of fUSI data

## Installation

1. Ensure you are on a macOS Apple Silicon or Ubuntu 22.04 x86_64 system.
2. Clone the Anise repository:

```console
git clone git@github.com:Forest-Neurotech/Anise.git
cd Anise
```

4. Set up a Python environment using [`uv`](https://astral.sh/uv), like so:
```console
# Set your Gemfury Forest-index token in the environment or in .env file
export UV_INDEX_FOREST_USERNAME=<your-token>
make install
source .venv/bin/activate
```