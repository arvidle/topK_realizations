# An efficient algorithm to produce the realizations of a stochastically known event log in ranked order

This repository provides an algorithm to efficiently calculate the realizations of a stochastically known event log, ordered by monotonically decreasing probability.

## Setup

First, ensure that Python >= 3.12 is installed.
Then, install the packages listed in requirements.txt with the package manager of your choice (e.g., pip/conda)
For the example in _run_on_xes.py_, the HR log from [https://github.com/HaifaUniversityBPM/traffic-data-to-event-log](https://github.com/HaifaUniversityBPM/traffic-data-to-event-log) in XES format has to be placed in a folder named _data_.
For the LaTeX-style axis labels, a local installation of LaTeX is required.

## Usage

### Manual Usage
To reproduce the evaluation results, simply run _sensitivity_analysis.py_.
Then, create the plots under _evaluation/results/plots_ by running _python plot\_sensitivity\_analysis.py "sensitivity\_analysis\_results.pkl"_.

The file _run_on_xes.py_ provides an example on how to apply the algorithm to uncertain event logs in XES format.

### Jupyter Notebook

Simply import _reproduce_eval.ipynb_ into Google Colab and run it.