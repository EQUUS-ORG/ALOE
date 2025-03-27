import math
import os
import sys
from typing import List

import pandas as pd
import psutil
import torch
from rdkit import Chem


def make_output_name(input_file, suffix, file_type):
    """
    Creates the output file with the form "<input_file>_<suffix><file_type>".
    Any suffix related to the output from a previous step is removed.

    Arguments:
        input_file: The path to the input file.
        suffix: The suffix to be added to the output file name.
        file_type: The type of the output file (e.g., ".sdf", ".csv").

    Returns:
        output_file: The path to the output file.
    """
    basename = os.path.basename(input_file).split(".")[0]

    basename_segs = basename.split("_")

    curr_suffix = basename_segs[-1]
    output_suffixes = ["isomers", "embedded", "opt", "ranked", "thermo"]
    if curr_suffix in output_suffixes:
        basename = "_".join(basename_segs[:-1])

    return os.path.join(os.path.dirname(input_file), f"{basename}_{suffix}{file_type}")


def smiles_from_file(input_file) -> List[str]:
    """
    Reads a file and returns a list of SMILES strings.
    """
    if input_file.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        smiles = [Chem.MolToSmiles(mol) for mol in filter(None, suppl)]
    elif input_file.endswith(".csv"):
        df = pd.read_csv(input_file, header=0, dtype={df.columns[1]: str})
        smiles = df.iloc[:, 1].tolist()

    return smiles


def check_input(input_file, expected_input_format):
    """
    Check the input file and give recommendations.

    Arguments:
        input_file: The path to the input file.

    Returns:
        This function checks the format of the input file, the properties for

    """
    print("Checking input file...", flush=True)  # Check the input format
    print("Input file:" + input_file, flush=True)
    if not input_file.endswith(expected_input_format):
        sys.exit(
            f"Input file must be in {expected_input_format} format. Please check the input file."
        )

    if expected_input_format == "csv":
        check_csv_format(input_file)
    elif expected_input_format == "sdf":
        check_sdf_format(input_file)
    print("Input file format is correct.", flush=True)


def check_sdf_format(input_file):
    """
    Check the input sdf file.

    Arguments:
        input_file: The path to the input file.

    Returns:
        True if the input file is valid, will not return otherwise

    """

    supp = Chem.SDMolSupplier(input_file, removeHs=False)
    mols = []
    for mol in filter(None, supp):
        id = mol.GetProp("_Name")
        assert len(id) > 0, "Empty ID"
        mols.append(mol)

    print(
        f"\tThere are {len(mols)} conformers in the input file {input_file}. ",
        flush=True,
    )
    print("\tAll conformers and IDs are valid.", flush=True)

    return True


def check_csv_format(input_file):
    """
    Checks the input file so that column 1 is unique names and column 2 is smiles strings

    Arguments:
        input_file: The path to the input file.

    Returns:
        True if the input file is valid, will not return otherwise
    """

    df = pd.read_csv(input_file, header=0, dtype={df.columns[1]: str})

    if df.shape[1] != 2:
        sys.exit("The input file should have two columns.")

    if df.dtypes[1] != object:
        sys.exit("The input file must have a second column of type string.")

    if df.iloc[:, 0].isna().sum() != 0 or df.iloc[:, 1].isna().sum() != 0:
        sys.exit("The input file should not have any missing values.")

    if df.iloc[:, 0].duplicated().sum() != 0:
        sys.exit("The input file should have unique names in the first column.")

    invalid_smiles = []

    for smiles in df.iloc[:, 1]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append(smiles)

    if len(invalid_smiles) != 0:
        sys.exit(f"The following smiles strings are invalid: {invalid_smiles}")

    print(
        f"\tThere are {len(df)} smiles in the input file {input_file}. ",
        flush=True,
    )
    print("\tAll smiles are valid.", flush=True)

    return True


def read_csv(input_file):
    """Reads a CSV file and returns the names and SMILES strings as lists."""
    df = pd.read_csv(input_file)

    # Extract columns by index
    names = df.iloc[:, 0]  # First column (col 0) = Names
    smiles = df.iloc[:, 1]  # Second column (col 1) = SMILES

    # Convert to list
    names = names.tolist()
    smiles = smiles.tolist()

    return names, smiles


def read_csv_dict(input_file):
    """Reads a CSV file and returns a dictionary with names as keys and SMILES strings as values."""
    df = pd.read_csv(input_file)

    # Extract columns by index
    names = df.iloc[:, 0]  # First column (col 0) = Names
    smiles = df.iloc[:, 1]  # Second column (col 1) = SMILES
    data_dict = dict(zip(names, smiles))
    return data_dict


def SDF2chunks(sdf: str) -> List[List[str]]:
    """given a sdf file, return a list of chunks,
    each chunk consists of lines of a molecule as they appear in the original file"""
    chunks = []
    with open(sdf, "r") as f:
        data = f.readlines()
        f.close()
    chunk = []
    for line in data:
        if line.strip() == "$$$$":
            chunk.append(line)
            chunks.append(chunk)
            chunk = []
        else:
            chunk.append(line)
    return chunks


def fill_chunks(path0, file_type):
    """Given a file path and file type, returns the indexes of the molecules in the file, ordered by size and groupped by name."""

    if file_type == "sdf":
        suppl = Chem.SDMolSupplier(path0, removeHs=False)
    elif file_type == "csv":
        _, smiles = read_csv(path0)
        suppl = []
        for smile in smiles:
            if smile is not None:
                mol = Chem.AddHs(Chem.MolFromSmiles(smile))
                mol.SetProp("_Name", smile)
                suppl.append(mol)

    # keys are the molecule names,
    # values are lists of index in path and number of atoms
    molecule_dict = {}
    for idx, mol in enumerate(suppl):
        if mol is not None:
            identifier = mol.GetProp("_Name").strip()

            if identifier not in molecule_dict:
                molecule_dict[identifier] = []
            mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()
            molecule_dict[identifier].append((idx, num_atoms))

    return batch(molecule_dict)


def batch(molecule_dict):
    """Given a dictionary of molecules, return a list of indexes of the molecules in the file, ordered by size and groupped by name."""

    # a list of the molecule indexes in the original file, sorted by size (ascending) and grouped by name
    ordered_indexes = []
    total_size = 0

    # a list of tuples of molecule identifiers and the total size of all their atoms
    before_ordered_indexes = []
    for identifier, same_name_list in molecule_dict.items():
        same_name_size = sum(num_atoms for _, num_atoms in same_name_list)
        before_ordered_indexes.append((identifier, same_name_size))
        total_size += same_name_size

    almost_sorted = sorted(before_ordered_indexes, key=lambda x: x[1])
    ordered_indexes = []
    for identifier, _ in almost_sorted:
        index_list = molecule_dict[identifier]
        for idx, _ in index_list:
            ordered_indexes.append(idx)

    return ordered_indexes


def _divide_jobs_based_on_memory(hardware_settings):
    """Helper function to divide jobs based on available memory and update GPU settings."""
    # Allow 42 SMILES strings per GB memory by default for generate_and_optimize_conformers

    capacity = hardware_settings["capacity"]
    memory = hardware_settings["memory"]
    use_gpu = hardware_settings["use_gpu"]
    gpu_idx = hardware_settings["gpu_idx"]
    batchsize_atoms = hardware_settings["batchsize_atoms"]

    smiles_per_G = capacity
    num_jobs = 1
    if memory is not None:
        t = int(memory)
    else:
        if use_gpu:
            if isinstance(gpu_idx, int):
                first_gpu_idx = gpu_idx
            else:
                first_gpu_idx = gpu_idx[0]
                num_jobs = len(gpu_idx)
            if torch.cuda.is_available():
                t = int(
                    math.ceil(
                        torch.cuda.get_device_properties(first_gpu_idx).total_memory
                        / (1024**3)
                    )
                )
        else:
            t = int(psutil.virtual_memory().total / (1024**3))
    chunk_size = t * smiles_per_G
    # batchsize_atoms based on GPU memory
    batchsize_atoms = batchsize_atoms * t
    return t, batchsize_atoms, num_jobs, chunk_size


def _save_chunks(input_file, t, num_jobs, chunk_size):
    r"""
    Given an input file, divide the file into chunks based on the available memory and the number of jobs.
    """

    basename = os.path.basename(input_file).split(".")[0].strip()
    input_format = input_file.split(".")[-1].strip()
    int_dir = os.path.join(os.path.dirname(input_file), basename + "_intermediates")
    os.mkdir(int_dir)

    ordered_indexes = fill_chunks(input_file, input_format)

    # placeholder
    if input_format == "csv":
        names, smiles = read_csv(input_file)
        data_size = len(smiles)

        current_write_index = 0
        num_chunks = max(round(data_size // chunk_size), num_jobs)

        print(f"The available memory is {t} GB.", flush=True)
        print(f"The task will be divided into {num_chunks} job(s).", flush=True)

        mols_per_chunk = math.ceil(data_size / num_chunks)

        chunked_files = []

        for i in range(num_chunks):
            new_basename = basename + "_" + str(i + 1) + f".{input_format}"
            new_name = os.path.join(int_dir, new_basename)
            curr_job_inputs = 0
            with open(new_name, "w") as f:
                f.write("Name,SMILES\n")
                while (
                    current_write_index % mols_per_chunk < mols_per_chunk
                    and current_write_index < data_size
                ):
                    sorted_index = ordered_indexes[current_write_index]
                    f.write(f"{names[sorted_index]},{smiles[sorted_index]}\n")
                    current_write_index += 1
                    curr_job_inputs += 1
            print(f"Job{i+1}, number of inputs: {curr_job_inputs}", flush=True)
            chunked_files.append(new_name)

    elif input_format == "sdf":
        # Get indexes for each chunk
        df = SDF2chunks(input_file)
        data_size = len(df)
        num_chunks = max(round(data_size // chunk_size), num_jobs)

        print(f"The available memory is {t} GB.", flush=True)
        print(f"The task will be divided into {num_chunks} job(s).", flush=True)

        current_write_index = 0
        mols_per_chunk = math.ceil(data_size / num_chunks)
        # Save each chunk as an individual file
        chunked_files = []
        for i in range(num_chunks):
            new_basename = basename + "_" + str(i + 1) + f".{input_format}"
            new_name = os.path.join(int_dir, new_basename)
            curr_job_inputs = 0
            with open(new_name, "w") as f:
                while (
                    current_write_index % mols_per_chunk < mols_per_chunk
                    and current_write_index < data_size
                ):
                    sorted_index = ordered_indexes[current_write_index]
                    for line in df[sorted_index]:
                        f.write(line)
                    current_write_index += 1
                    curr_job_inputs += 1
            print(f"Job{i+1}, number of inputs: {curr_job_inputs}", flush=True)
            chunked_files.append(new_name)

    return chunked_files


def combine_files(files, input_file, output_dir, output_suffix="_out.sdf"):
    """Combine multiple files into a single file."""
    if len(files) == 0:
        msg = """The optimization engine did not run, or no 3D structure converged.
                 The reason might be one of the following:
                 1. Allocated memory is not enough;
                 2. The input SMILES encodes invalid chemical structures;
                 3. Patience is too small."""
        sys.exit(msg)

    data = []
    for file in files:
        with open(file, "r") as f:
            data_i = f.readlines()
        data += data_i
    suffix, output_type = output_suffix.split(".")
    output_type = "." + output_type
    basename = os.path.basename(input_file).split(".")[0]
    path_combined = os.path.join(output_dir, f"{basename}{suffix}{output_type}")
    with open(path_combined, "w+") as f:
        for line in data:
            f.write(line)
    return path_combined


def _print_timing(start, end):
    print("Energy unit: Hartree if implicit.", flush=True)
    running_time_m = int((end - start) / 60)
    if running_time_m <= 60:
        print(f"Program running time: {running_time_m + 1} minute(s)", flush=True)
    else:
        running_time_h = running_time_m // 60
        remaining_minutes = running_time_m - running_time_h * 60
        print(
            f"Program running time: {running_time_h} hour(s) and {remaining_minutes} minute(s)",
            flush=True,
        )
