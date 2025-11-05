import os
import sys
import warnings

import torch
from rdkit import Chem

from .file_utils import read_csv

# CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026

ALLOWED_ENGINES = {"ANI2x", "ANI2xt", "AIMNET", "AIMNET-lite"}


def check_shared_parameters(opt_config, thermo_config):
    r"""
    Check if the shared parameters between optimization and thermochemistry are consistent.
    Args:
        opt_config: The optimization configuration.
        thermo_config: The thermochemistry configuration.
    """
    if opt_config["optimizing_engine"] != thermo_config["model_name"]:
        warnings.warn("The engine for optimization and thermochemistry do not match.")


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
        if torch.cuda.is_available():
            device = f"cuda:{gpu_idx}"
            print(f"Using GPU {gpu_idx}.")
        else:
            sys.exit("No GPU was detected. Please set --use_gpu=False.")
    else:
        device = "cpu"
        print("Using CPU.")
    return device


def check_model(model, input_file):
    r"""
    Check if the model is valid and compatible with the input file.
    Args:
        model: The model to be used (ANI2x, ANI2xt, AIMNET-lite, or a path to a userNNP model).
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
    if model == "AIMNET-lite":
        validate_aimnet(input_file)
    elif model == "UMA":
        # UMA supports broad element/charge coverage
        pass
    else:
        sys.exit(
            f"Unknown engine '{model}'. Allowed engines: {sorted(ALLOWED_ENGINES)} or provide a path to a TorchScript user NNP."
        )


def validate_ANI2x(input_file):
    r"""
    Checks if molecules in the input file are compatible with ANI2x.
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}

    file_type = input_file.split(".")[-1].strip()
    if file_type == "sdf":
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        invalid_ani_smiles = []
        for mol in suppl:
            if mol is not None:
                charge = Chem.rdmolops.GetFormalCharge(mol)
                elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
                if (elements.issubset(ANI_elements) is False) or (charge != 0):
                    invalid_ani_smiles.append(Chem.MolToSmiles(mol))
    elif file_type == "csv":
        _, smiles = read_csv(input_file)
        invalid_ani_smiles = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            charge = Chem.rdmolops.GetFormalCharge(mol)
            elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
            if (elements.issubset(ANI_elements) is False) or (charge != 0):
                invalid_ani_smiles.append(smi)

    if len(invalid_ani_smiles) != 0:
        sys.exit(
            f"The following smiles strings are incompatible with ANI: {invalid_ani_smiles}"
        )


def validate_aimnet(input_file):
    r"""
    Checks if molecules in the input file are compatible with AIMNET-lite.
    """
    # Ensures that molecules are compatible with AIMNET
    # AIMNET elemets = H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
    AIMNET_elements = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}

    file_type = input_file.split(".")[-1].strip()
    if file_type == "sdf":
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        invalid_aimnet_smiles = []
        for mol in suppl:
            if mol is not None:
                # Only singlet states allowed
                multiplicity = (
                    sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) + 1
                )
                elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
                if (elements.issubset(AIMNET_elements) is False) or (multiplicity != 1):
                    invalid_aimnet_smiles.append(Chem.MolToSmiles(mol))

    elif file_type == "csv":
        _, smiles = read_csv(input_file)
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
            f"The following smiles strings are incompatible with AIMNET-lite: {invalid_aimnet_smiles}"
        )
