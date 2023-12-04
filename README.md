# Michaelis-Menten equation fitting to kinetic data

A panel app that implements five methods to fit the Michaelis-menten equation to a set of points of rate vs substrate concentration.

The methods are:

- Non-linear regression (Levemberg-Marquard algorithm applied to Michaelis-Menten equation)
- The Direct Linear Plot
- The Lineweaver-Burk linearization
- The Hanes linearization
- The Eddie-Hofstee linearization

## mm_fitting

This repo includes also code for `mm_fitting`, a supporting module that implements the five methods, which is "pip installable" and can be imported in third-party code (MIT licence).

## Installation and running the app

Creating a `conda` [environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to run the app is recommended:

1. Git clone the repo, and `cd` into the folder (at an Anaconda or Miniconda Terminal/shell/Powershell)
2. Run `conda env create -f mmhyperfit.yml` to create the environment.

To run the app:

1. `cd` into the app folder (at an Anaconda or Miniconda Terminal/shell/Powershell)
2. Run `conda activate mmhyperfit`
3. Run `panel serve app.py --show`

## Demo notebook

Alernatively, run the notebook `demo.ipynb``. It can be used to produce all tables and plots. Just replace the demo data with your own.

