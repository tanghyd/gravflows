### ozstar module loads + virtualenv
`module load python/3.8.5`

We could replace later `pip install`s with `module load`, e.g.

`module load wheel/0.37.0-python-3.8.5`

We do load wheel as above, but the rest is downloaded via pip.

## Virtualenv install commands

    python -m venv <name>
    . <name>/bin/activate

## Conda install commands

    conda create --name <name> python=3.8.5 -y
    conda activate <name>  # source activate <name> works for containers

## Shared install commands

Upgrade pip inside conda (or just in regular virtual environment)

`python -m pip install --upgrade pip`

Install main packages

`pip install tqdm numpy scipy pandas scikit-learn matplotlib seaborn pycbc lalsuite ligo.skymap`  # core
`pip install wheel black pytest mypy ipykernel`

OR

`pip install -r requirements.txt`

## Miscellaneous notes

Sometimes `astropy.utils.iers` will call something to auto-download data.
As compute nodes have no internet access, we have disabled this in `gwpe/waveforms.py` with:

    from astropy.utils import iers
    iers.conf.auto_download = False