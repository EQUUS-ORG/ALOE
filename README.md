# <img src="pics/aloe.png" width="24"/> ALOE: Asynchronous Lightweight Optimization Engine


## What is this?

ALOE is a simple pipeline that generates and optimizes conformers, using a neural interatomic potential as the calculator. A complete end-to-end workflow generates optimized 3d conformers from SMILES strings, with electronic and Gibbs free energies evaluated.

<p align="center">
    <img src="pics/ALOE_flowchart.png" width="1000" style="display: block"/>
<p>
The backend is adapted from Auto3D (https://github.com/isayevlab/Auto3D_pkg). The default model is AIMNet2 (https://chemrxiv.org/engage/chemrxiv/article-details/6763b51281d2151a022fb6a5).

ALOE's front-end grants full control over individual operations. Please see below for an example that includes all the steps shown in the previous flow chart.

```python3
import aloe

engine = aloe.aloe(input_file = "test.csv")
engine.add_step(aloe.StereoIsoConfig()) # Generate stereoisomers
engine.add_step(aloe.ConformerConfig()) # Embed conformers
engine.add_step(aloe.OptConfig()) # Optimize conformers
engine.add_step(aloe.RankConfig(k=3)) # Rank optimized conformers, pick the best 3
engine.add_step(aloe.ThermoConfig()) # Thermochemistry calculations via ASE
output_file = engine.run() # Asynchronous execution

print(output_file)
```

## Installation

We recommend creating a virtual environment first. 
```bash
conda create -n aloe python=3.12 -y
conda activate aloe
```

Install PyTorch. 

```bash
pip install torch
```

To install ALOE in editable mode:
```bash
cd to/this/directory
pip install -e .
```

Or simply
```bash
pip install aloe
```

## Why aynchronous execution?

Molecules in the input files are batched at the start of the job according to their sizes (numbers of atoms) and the system's memory (RAM) limit. All subsequent steps are executed asynchronously to optimize usage of available CPUs/GPUs as specified by the user.

## Citations

Please consider citing the original Auto3D paper if you find ALOE helpful. 

```
@article{
    liu2022auto3d,
    title={Auto3d: Automatic generation of the low-energy 3d structures with ANI neural network potentials},
    author={Liu, Zhen and Zubatiuk, Tetiana and Roitberg, Adrian and Isayev, Olexandr},
    journal={Journal of Chemical Information and Modeling},
    volume={62},
    number={22},
    pages={5373--5382},
    year={2022},
    publisher={ACS Publications}
}
```

To be filled.
