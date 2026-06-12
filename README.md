<div align="center">

<img src="pics/aloe.png" width="64"/>

# ALOE

### Adaptive Lightweight Optimization Engine

**SMILES in. Optimized 3D conformers out.**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ready-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Model-AIMNet2-8A2BE2)](https://chemrxiv.org/engage/chemrxiv/article-details/6763b51281d2151a022fb6a5)

</div>

---

ALOE is a simple pipeline that generates and optimizes conformers, using a neural interatomic potential as the calculator. A complete end-to-end workflow generates optimized 3D conformers from SMILES strings, with electronic and Gibbs free energies evaluated. Works best on a Linux/Mac operating system.

<p align="center">
  <img src="pics/ALOE_flowchart.png" width="1000"/>
</p>

The backend is adapted from [Auto3D](https://github.com/isayevlab/Auto3D_pkg "Auto3D GitHub Repository"). The default model is [AIMNet2](https://chemrxiv.org/engage/chemrxiv/article-details/6763b51281d2151a022fb6a5).

ALOE's front-end grants full control over individual operations. Please see below for an example that includes all the steps shown in the previous flow chart.

```python
import aloe

if __name__ == "__main__":
    engine = aloe.aloe(input_file = "test.csv")
    engine.add_step(aloe.StereoIsoConfig()) # Generate stereoisomers
    engine.add_step(aloe.ConformerConfig()) # Embed conformers
    engine.add_step(aloe.OptConfig())       # Optimize conformers, add argument use_gpu=True to use GPU
    engine.add_step(aloe.RankConfig(k=3))   # Rank optimized conformers, pick the best 3
    engine.add_step(aloe.ThermoConfig())    # Thermochemistry calculations via ASE
    output_file = engine.run()             # Concurrent execution

    print(output_file)
```

---

## Installation

We recommend creating a virtual environment first.

```bash
conda create -n aloe python=3.12 -y
conda activate aloe
```

Install PyTorch based on your operating system, for reference visit [PyTorch Installation](https://pytorch.org/get-started/locally/). For Mac users, use the following command.

```bash
pip install torch
```

Then install ALOE. Choose either:

```bash
# From source (editable)
cd to/this/directory
pip install -e .
```

```bash
# From PyPI
pip install aloe-engine
```

---

## Why asynchronous execution?

Molecules in the input files are batched at the start of the job according to their sizes (numbers of atoms) and the system's memory (RAM) limit. All subsequent steps are executed concurrently to optimize usage of available CPUs/GPUs as specified by the user.

---

## Citations

Please consider citing the original Auto3D paper if you find ALOE helpful.

```bibtex
@article{liu2022auto3d,
    title     = {Auto3d: Automatic generation of the low-energy 3d structures with ANI neural network potentials},
    author    = {Liu, Zhen and Zubatiuk, Tetiana and Roitberg, Adrian and Isayev, Olexandr},
    journal   = {Journal of Chemical Information and Modeling},
    volume    = {62},
    number    = {22},
    pages     = {5373--5382},
    year      = {2022},
    publisher = {ACS Publications}
}
```

*ALOE-specific citation to be filled.*