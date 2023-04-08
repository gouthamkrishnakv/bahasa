# Python OpenCV Simple Start

## Install Virtual Environment (`virtualenv`) for project

```bash
# Install Virtualenv (if not installed)
pip install -U virtualenv
# Create a virtual environment in `.venv` path.
virtualenv -p python3.9 .venv
# Enable Virtual Environment: Windows ONLY
. .venv/Scripts/activate
# Enable Virtual Environment: Linux/macOS ONLY
source .venv/bin/activate
# Install all dependencies
pip install -r requirements.txt
```

## Run the application

In the same shell run

```sh
# For Linux/macOS. Check whether the path returns `python` inside `.venv`.
# Should return ".../bahasa/.venv/bin/python"
which python
# For Windows. 
# Path should be ".../bahasa/.venv/bin/python"
Get-Command python
# Run the application (runs `__main__.py` inside `bahasa`)
python -m bahasa
```