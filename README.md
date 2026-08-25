# The Effect of Expertise Signaling when Sharing Medical Information in Opinion Dynamics
Here we provide our code for the data-driven agent-based model in our paper ('The Effect of Expertise Signaling when Sharing Medical Information in Opinion Dynamics').

## Introduction
In the Online Experiment folder, we provide the cleaned-up data we collected from the online experiment and analysis on the independent variable Title.
In the Simulation folder, we provide the agent-based model and our simulations. The simulations for E1 and E2 are separately provided in Jupyter notebooks distinguished by network types (`ER_26.ipynb`, `SF_26.ipynb`, `PBN_26.ipynb`). The simulations for E3 and E4 are provided together in `E3_E4.ipynb`.

## Requirements
This project uses the following Python packages:
- `numpy` – for numerical operations
- `pandas` – for data analysis
- `matplotlib` – for plotting results
- `networkx` - for building the environment
- `scipy` – used for statistical tests and distributions (`scipy.stats`)
- `seaborn` – for creating heatmaps

To install the external packages, run:

```bash
pip install scipy seaborn

## Additional Experiments and Results
With the same agent-based model, we conducted four simulation experiments in total. Description and results of Experiment 2-4 are included in the attached file as an extension of the current work.
