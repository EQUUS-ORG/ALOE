import sys
import time
from typing import Callable, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from aloe.ASE import thermo
from aloe.batch_opt.batchopt import optimizing
from aloe.embed import embedding
from aloe.file_utils import _print_timing, check_input, make_output_name, read_csv
from aloe.isomer_generation.isomer_engine import rd_enumerate_isomer
from aloe.model_validation import check_device, check_model
from aloe.rank.ranking import ranking

max_conformers_per_GB_memory = 8192


def generate_stereoisomers(
    input_file: str,
    enumerate_tautomers: Optional[bool] = False,
    onlyUnassigned: Optional[bool] = True,
    unique: Optional[bool] = True,
) -> str:
    r"""
    Takes in a csv file with SMILES strings and generates stereoisomers for each molecule in an output csv file.
    Args:
        input_file (str): path to the input csv file containing the SMILES strings.
        enumerate_tautomers (bool, optional): Whether to enumerate tautomers for input. Defaults to False.
        onlyUnassigned (bool, optional): Whether to enumerate only unassigned tautomers. Defaults to True.
        unique (bool, optional): Whether to enumerate unique tautomers. Defaults to True.
    Returns:
        output_file: path to the output sdf file containing the generated stereoisomers.
    """

    io_file_type = ".csv"

    check_input(input_file, io_file_type)
    output_file = make_output_name(input_file, "isomers", io_file_type)
    print("Generating stereoisomers...", flush=True)

    engine = rd_enumerate_isomer(
        csv=input_file,
        enumerated_csv=output_file,
        enumerate_tauts=enumerate_tautomers,
        onlyUnassigned=onlyUnassigned,
        unique=unique,
    )

    output_file = engine.run()
    print("Finished generating stereoisomers.", flush=True)
    return output_file


def embed_conformers(
    input_file: str,
    max_conformers: Optional[int] = None,
    mpi_np: Optional[int] = 4,
    threshold: Optional[float] = 0.3,
) -> str:
    r"""
    Takes in a csv file with SMILES strings and embeds conformers for each molecule in an output sdf file.
    Args:
        input_file (str): path to the input csv file containing the SMILES strings.
        max_conformers (int, optional): maximum number of conformers to generate for each SMILES.
        mpi_np (int, optional): Number of CPU cores for isomer generation. Defaults to 4.
        threshold (float, optional): RMSD threshold for considering conformers as duplicates. Defaults to 0.3


    Returns:
        output_file: path to the output sdf file containing the embedded conformers.
    """

    input_file_type = ".csv"
    output_file_type = ".sdf"

    check_input(input_file, input_file_type)
    output_file = make_output_name(input_file, "embedded", output_file_type)
    print("Embeding conformers...", flush=True)

    # Writing to output path
    with Chem.SDWriter(output_file) as writer:
        names, smiles = read_csv(input_file)
        for name, smile in zip(names, smiles):
            mol = embedding.embed_conformer(smile, max_conformers, mpi_np, threshold)
            for i in range(mol.GetNumConformers()):
                positions = mol.GetConformer(i).GetPositions()
                # atoms clash if min distance is smaller than 0.9 Angstrom
                if embedding.min_pairwise_distance(positions) < 0.9:
                    AllChem.MMFFOptimizeMolecule(mol, confId=i)
                positions = mol.GetConformer(i).GetPositions()
                if embedding.min_pairwise_distance(positions) > 0.9:
                    conf_id = name.strip() + f"_{i}"
                    mol.SetProp("ID", conf_id)
                    mol.SetProp("_Name", name)
                    writer.write(mol, confId=i)

    print("Finished embedding conformers.", flush=True)

    return output_file


def optimize_conformers(
    input_file: str,
    capacity: Optional[int] = max_conformers_per_GB_memory,
    opt_use_gpu: Optional[bool] = True,
    gpu_idx: Optional[int] = 0,
    batchsize_atoms: Optional[int] = 2048,
    optimizing_engine: Optional[str] = "AIMNET",
    patience: Optional[int] = 1000,
    opt_steps: Optional[int] = 5000,
    convergence_threshold: Optional[float] = 0.003,
) -> str:
    r"""
    Takes in a sdf file with conformers and optimizes the geometry of each conformer into an output sdf file.
    Arguments:
        input_file (str): path to the input sdf file containing conformers.

        ## Hardware Settings
        capacity (int, optional): number of molecules to process per 1GB memory. Defaults to 8192.
        use_gpu (bool, optional): Whether to use GPU when available. Defaults to True.
        gpu_idx (int, optional): GPU index to use. Only applies when use_gpu=True. Defaults to 0.
        batchsize_atoms (int, optional): Number of atoms per optimization batch per 1GB. Defaults to 2048.

        ## Optimization Parameters
        optimizing_engine (str, optional): Geometry optimization engine.
                                        Choose from 'ANI2x', 'ANI2xt', 'AIMNET' or path to custom NNP. Defaults to "AIMNET".
        patience (int, optional): Maximum consecutive steps without force decrease before termination. Defaults to 1000.
        opt_steps (int, optional): Maximum optimization steps per structure. Defaults to 5000.
        convergence_threshold (float, optional): Maximum force threshold for convergence. Defaults to 0.003.

    Returns:
        output_file: path to the output sdf file containing the optimized conformers.
    """

    io_file_type = ".sdf"
    # Checks input file, device, and model
    check_input(input_file, io_file_type)
    device = check_device(opt_use_gpu, gpu_idx)
    check_model(optimizing_engine, input_file)
    print("Optimizing Conformers...", flush=True)

    if int(opt_steps) < 10:
        sys.exit(
            f"Number of optimization steps cannot be smaller than 10, but received {opt_steps}"
        )

    output_file = make_output_name(input_file, "opt", io_file_type)

    if capacity < max_conformers_per_GB_memory:
        print(
            f"The capacity (number of molecules per 1GB memory) is too small. Please set it to a value >= {max_conformers_per_GB_memory}.",
            flush=True,
        )

    start = time.time()

    # Assign device

    opt_config = {
        "opt_steps": opt_steps,
        "opttol": convergence_threshold,
        "patience": patience,
        "batchsize_atoms": batchsize_atoms,
    }

    optimizer = optimizing(
        in_f=input_file,
        out_f=output_file,
        name=optimizing_engine,
        device=device,
        config=opt_config,
    )
    optimizer.run()

    end = time.time()
    print("Finished optimizing conformers.", flush=True)
    _print_timing(start, end)

    return output_file


def rank_conformers(
    input_file: str,
    k: Optional[int] = None,
    window: Optional[bool] = None,
    threshold: Optional[float] = 0.3,
) -> str:
    r"""
    Takes in a sdf file with conformers and ranks the conformers based on their energy into an output sdf file.
    Args:
        input_file (str): path to the input sdf file containing conformers.
        k (int, optional): Number of conformers for each molecule. Defaults to None.
        window (bool, optional): Whether to output structures with energies within x kcal/mol from the lowest energy conformer. Defaults to None.
        threshold (float, optional): RMSD threshold for considering conformers as duplicates. Defaults to 0.3.

    Returns:
        output_file: path to the output sdf file containing the ranked conformers.
    """

    io_file_type = ".sdf"

    check_input(input_file, io_file_type)
    print("Ranking Conformers...", flush=True)

    if k is None and window is None:
        sys.exit("Either k or window must be provided for ranking conformers. ")

    # Output path
    output_file = make_output_name(input_file, "ranked", io_file_type)

    rank_engine = ranking(
        input_path=input_file,
        out_path=output_file,
        threshold=threshold,
        k=k,
        window=window,
        encoded=True,
    )

    output_file = rank_engine.run()
    print("Finished ranking conformers.", flush=True)
    return output_file


def calculate_thermo(
    input_file: str,
    model_name: str = "AIMNET",
    mol_info_func: Optional[Callable] = None,
    thermo_use_gpu: Optional[bool] = True,
    gpu_idx: int = 0,
    opt_tol: float = 0.0002,
    opt_steps: int = 5000,
) -> str:
    r"""
    Takes in a sdf file with conformers and calculates the thermochemical properties of each conformer into an output sdf file.
    Args:
        input_file (str): path to the input sdf file.

        ## Hardware Settings
        use_gpu (bool, optional): Whether to use GPU when available. Defaults to True.
        gpu_idx (int, optional): GPU index to use. Only applies when use_gpu=True. Defaults to 0.

        ## Thermal Calculation Parameters
        model_name (str, optional): name of the forcefield to use. Defaults to "AIMNET".
        mol_into_function (Callable, optional): function to convert the molecule into a format that can be used by the forcefield. Defaults to None.
        opt_tol (float, optional): Convergence_threshold for geometry optimization. Defaults to 0.0002.
        opt_steps (int, optional): Maximum optimization steps per structure. Defaults to 5000.

    Returns:
        output_file: path to the output csv file containing the calculated thermochemical properties.
    """

    io_file_type = ".sdf"

    check_input(input_file, io_file_type)
    check_device(thermo_use_gpu, gpu_idx)
    check_model(model_name, input_file)
    print("Thermo Calculations...", flush=True)

    # Output path
    output_file = make_output_name(input_file, "thermo", io_file_type)
    output_file = thermo.calc_thermo(
        input_file=input_file,
        output_file=output_file,
        model_name=model_name,
        mol_info_func=mol_info_func,
        use_gpu=thermo_use_gpu,
        gpu_idx=gpu_idx,
        opt_tol=opt_tol,
        opt_steps=opt_steps,
    )
    print("Finished thermo calculations.", flush=True)
    return output_file
