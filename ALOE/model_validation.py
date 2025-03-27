import os
import sys
import warnings

import torch
from rdkit import Chem

from aloe.file_utils import smiles_from_file

# CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026


def check_shared_parameters(opt_config, thermo_config):
    r"""
    Check if the shared parameters between optimization and thermochemistry are consistent.
    Args:
        opt_config: The optimization configuration.
        thermo_config: The thermochemistry configuration.
    """
    if opt_config["opt_use_gpu"] != thermo_config["thermo_use_gpu"]:
        warnings.warn(
            "The GPU settings for optimization and thermochemistry do not match."
        )

    if opt_config["optimizing_engine"] != thermo_config["model_name"]:
        warnings.warn("The engine for optimization and thermochemistry do not match.")

    if opt_config["memory"] != thermo_config["memory"]:
        sys.exit(
            "The GPU settings for optimization and thermochemistry do not match. Please set them to the same value."
        )

    if opt_config["opt_gpu_idx"] != thermo_config["thermo_gpu_idx"]:
        sys.exit(
            "The GPU indices for optimization and thermochemistry do not match. Please set them to the same value."
        )


def check_device(use_gpu, gpu_idx):
    """
    Check if the GPU is available and set the device accordingly.

    Arguments:
        use_gpu: Whether to use GPU.
        gpu_idx: The index of the GPU to use.

    Returns:
        device: The device to be used.
    """
    if use_gpu:
        if torch.cuda.is_available() or torch.mps.is_available():
            device = torch.device(f"cuda:{gpu_idx}")
            print(f"Using GPU {gpu_idx}.")
        else:
            sys.exit("No GPU was detected. Please set --use_gpu=False.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def check_model(model, input_file):
    r"""
    Check if the model is valid and compatible with the input file.
    Args:
        model: The model to be used (ANI2x, ANI2xt, AIMNET, or a path to a userNNP model).
        input_file: The input file containing the molecules.
    """
    # Check the installation for open toolkits, torchani
    if model == "ANI2x":
        try:
            import torchani
        except:
            sys.exit("ANI2x is used as the engine, but TorchANI is not installed.")
    if os.path.exists(model):
        try:
            _ = torch.jit.load(model)
        except:
            sys.exit(
                "A path to a user NNP is used as the engine, but it cannot be loaded by torch.load. See this link for information about saving and loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model"
            )

    # Checks validity of the input file in context of the model
    if model in {"ANI2x", "ANI2xt"}:
        validate_ANI2x(input_file)
    if model == "AIMNET":
        validate_aimnet(input_file)


def validate_ANI2x(input_file):
    r"""
    Checks if molecules in the input file are compatible with ANI2x.
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    smiles = smiles_from_file(input_file)
    invalid_ani_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if (elements.issubset(ANI_elements) is False) or (charge != 0):
            invalid_ani_smiles.append(id)

    if len(invalid_ani_smiles) != 0:
        sys.exit(
            f"The following smiles strings are incompatible with AIMNET: {invalid_ani_smiles}"
        )


def validate_aimnet(input_file):
    r"""
    Checks if molecules in the input file are compatible with AIMNET.
    """
    # Ensures that molecules are compatible with AIMNET
    # AIMNET elemets = H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
    AIMNET_elements = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}
    smiles = smiles_from_file(input_file)
    invalid_aimnet_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Only singlet states allowed
            multiplicity = (
                sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) + 1
            )
            elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
            if (elements.issubset(AIMNET_elements) is False) or (multiplicity != 1):
                invalid_aimnet_smiles.append(smi)

    if len(invalid_aimnet_smiles) != 0:
        sys.exit(
            f"The following smiles strings are incompatible with AIMNET: {invalid_aimnet_smiles}"
        )
