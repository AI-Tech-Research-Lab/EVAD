## EVAD: Encrypted Vibrational Anomaly Detection With Homomorphic Encryption
This folder contains the data, scripts, and notebooks used for the experimental part of the paper "EVAD: Encrypted Vibrational Anomaly Detection With Homomorphic Encryption".

TL;DR: give a look at `Plots.pdf` and `Memory.pdf` to check the results written in the paper.

In more detail, this folder contains the following files and folders:

1. `data`: contains the Triaxial Bearing dataset's files (not on Git).
2. `results`: contains the files produced by `EVAD.ipynb`, which are tabular files with the statistics on the experiments.
3. `he_svm.py`: contains the Python functions needed by the EVAD pipeline.
4. `EVAD.ipynb`: contains the Jupyter Notebook in which the OCSVM is trained, and is tested on both plain and encrypted data. It saves the statistics of the computations in `results`.
5. `Plots.ipynb`: check this notebook to see that the results of the plain and encrypted processings are the same, as well as the time needed for the computations.
6. `Memory.ipynb`: check this notebook to see the memory required by the encrypted computation.
7. `DataExploration.ipynb`: check this notebook for an introduction to the considered dataset.
8. `times.csv`: used for the times histogram in the paper, created by `Plots.pdf`.

Moreover, to make it possible to replicate these experiments, we provide:
1. `pyproject.toml` and `poetry.lock`: you can use [Poetry](https://python-poetry.org/docs/) to create and manage a Python virtual environment with all the needed libraries installed;
2. `requirements.txt`, if you prefer to manage the dependencies manually. You can create a virtual environment, and install the libraries using `pip`.
3. `get_data.sh`, you can run this script to download the dataset.

For the easiness of reading, the notebooks `Plots.ipynb` and `Memory.ipynb` are also exported as PDF.
