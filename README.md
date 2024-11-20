# Bootstrapper

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Citation](#citation)
- [Funding](#funding)


A toolkit for bootstrapping and refining 3D instance segmentations and models from sparse 2D labels. 

- 🖥️ **CLI**: Command-line interface for training, prediction, post-processing
- ⚙️ **Configurable Models**: Flexible and hackable models using config files
- 🧱 **Blockwise Processing**: Efficient handling of large volumes

## Installation

To install `bootstrapper`, we first recommend creating a new conda environment:
```
conda create -n bs -c conda-forge python=3.12 graph-tool boost
conda activate bs
```
Then, install bootstrapper with:
```
pip install git+https://github.com/ucsdmanorlab/bootstrapper.git
```

## Getting Started

Bootstrapper has the following commands, typically run in the given order:
- `bs prepare` : Prepare data and config files for the following steps

- `bs train` : Train a model

- `bs predict` : Run inference on a volume

- `bs segment` : Segment affinities

- `bs evaluate` : Evaluate segmentations against ground truth or model predictions

- `bs filter` : Refine segmentations to create pseudo-ground truth

A **round** is a cycle of the above commands.
- Use `bs prepare` to create config files for one or multiple rounds.
- Refined segmentations from one round become training labels for the next round.

It also has:

- `bs run`: Runs the appropriate command for the given config file.
- `bs view` : A wrapper for `neuroglancer -d`
- `bs utils` offers functions for data manipulation and preprocessing.

## Examples
* [CREMI](examples/cremi)

## Citation
For questions about the preprint or this repository, please contact vvenu@utexas.edu

If you find Bootstrapper useful in your research, please consider citing our **[preprint](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v1)**:
```
@article {Thiyagarajan2024.06.14.599135,
	author = {Thiyagarajan, Vijay Venu and Sheridan, Arlo and Harris, Kristen M. and Manor, Uri},
	title = {A deep learning-based strategy for producing dense 3D segmentations from sparsely annotated 2D images},
	year = {2024},
	doi = {10.1101/2024.06.14.599135},
	URL = {https://www.biorxiv.org/content/early/2024/06/15/2024.06.14.599135},
}
```

## Funding 
Chan-Zuckerberg Imaging Scientist Award DOI https://doi.org/10.37921/694870itnyzk from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation (funder DOI 10.13039/100014989). 

NSF NeuroNex Technology Hub Award (1707356), NSF NeuroNex2 Award (2014862)

![image](https://github.com/ucsdmanorlab/bootstrapper/assets/64760651/4b4a6029-e1ba-42bb-ab8b-d9357cc46239)
