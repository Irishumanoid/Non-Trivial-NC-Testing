# Non-Trivial-NC-Testing
Simulations for our NC for Smart Cities project

### To create a Lava-compatible virtual environment
```bash
python3.10 -m venv lava_env
source lava_env/bin/activate
```
Then install Lava, etc.

To make the virtual environment accessible to ipynb files, within the right venv,
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=lava_env --display-name "Python 3.10 (lava_env)"
```

Select the correct kernel name at the top of the ipynb file