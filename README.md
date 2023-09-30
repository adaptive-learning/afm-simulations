# AFM simulations

This repo contains the code used in PhD thesis Exploring Statistical Biases in Educational Data: A Simulation-Based
Approach. It consists of the following four main part.

1. Custom implementation of Additive Factors Model (AFM).
2. Implementation of simulation framework.
3. Description of scenarios in JSON format.
4. Analysis scripts that produce various figures.  

## How to use

This repository is also a Python package managed by [Poetry](https://python-poetry.org/). The easiest way to run
the code is to install poetry package

```commandline
pip install poetry
```

install the package

```commandline
poetry install
```

and run one of the available scripts 

```commandline
poetry run figure-X-Y
```

that reproduce figures from the thesis.

## Available scripts

### Parameter stability
* Scripts `figure-3-2` and `figure-3-3` reproduce heatmaps from section about stability of estimated AFM parameters. Some
of these heatmaps are shown in Figure 3.2 and Figure 3.3.

### Effects of cheating

* Script `figure-4-1` reproduces scatter plots of estimated β and γ parameters with increasing number of cheating
students shown in Figure 4.1.
* Script `figure-4-2` reproduces histograms with estimate student α parameters as seen in Figure 4.2.
* Script `figure-4-3` reproduces learning curves shown in Figure 4.3.

### Effects of item ordering
* Script `figure-5-3` reproduces scatter plots with estimated β and γ parameters under fixed and random item ordering
depicted in Figure 5.3.
* Script `figure-5-4` reproduces barplots with various performance metrics comparing performance of AFM with correct
and misspecified Q-matrix from Figure 5.4.
* Script `figure-5-5` reproduces scatter plots showing detailed estimated values of β and γ parameters for AFM with
correct and misspecified Q-matrix from Figure 4.2.