import csv
import math
import os
import sys
import shutil
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


def read_csv(input_file):
    """Reads a CSV file and returns the names and SMILES strings as lists."""
    df = pd.read_csv(input_file, dtype=str)

    # Extract columns by index
    names = df.iloc[:, 0]  # First column (col 0) = Names
    smiles = df.iloc[:, 1]  # Second column (col 1) = SMILES

    # Convert to list
    names = names.tolist()
    smiles = smiles.tolist()

    return names, smiles


def read_csv_dict(input_file):
    """Reads a CSV file and returns a dictionary with names as keys and SMILES strings as values."""
    df = pd.read_csv(input_file, dtype=str)

    # Extract columns by index
    names = df.iloc[:, 0]  # First column (col 0) = Names
    smiles = df.iloc[:, 1]  # Second column (col 1) = SMILES
    data_dict = dict(zip(names, smiles))
    return data_dict


def check_input(input_file, expected_input_format):
    """
    Check the input file and give recommendations.

    Arguments:
        input_file: The path to the input file.

    Returns:
        This function checks the format of the input file, the properties for

    """
    print(
        f"Checking intermediate file {input_file}", flush=True
    )  # Check the input format
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
    num_mols = 0
    for mol in filter(None, supp):
        id = mol.GetProp("_Name")
        assert len(id) > 0, "Empty ID"
        num_mols += 1

    print(
        f"\tThere are {num_mols} conformers in the input file {input_file}. ",
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

    df = pd.read_csv(input_file, header=0, dtype=str)

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


def update_hardware_settings(hardware_settings: dict):
    """Helper function to divide jobs based on available memory and update GPU settings."""

    memory = hardware_settings["memory"]
    use_gpu = hardware_settings["use_gpu"]
    gpu_idx = hardware_settings["gpu_idx"]
    batchsize_atoms = hardware_settings["batchsize_atoms"]

    configured_memory_gb = 1
    if memory is not None:
        configured_memory_gb = int(memory)
    else:
        if use_gpu:
            if isinstance(gpu_idx, int):
                first_gpu_idx = gpu_idx
            else:
                first_gpu_idx = gpu_idx[0]
            if torch.cuda.is_available():
                configured_memory_gb = int(
                    math.ceil(
                        torch.cuda.get_device_properties(first_gpu_idx).total_memory
                        / (1024**3)
                    )
                )
        else:
            configured_memory_gb = int(psutil.virtual_memory().total / (1024**3))

    # batchsize_atoms based on GPU memory
    batchsize_atoms = batchsize_atoms * configured_memory_gb
    return configured_memory_gb, batchsize_atoms


def sdf2mols(sdf: str, indicies: List[int] = None) -> dict:
    """
    Given a sdf file, return a dictionary of mols, corresponding to the given indicies.
    If indicies is not provided, return all mols.
    Args:
        sdf: The path to the sdf file.
        indicies: The indicies of the molecules to return.
    Returns:
        A dictionary of mols.
    """
    suppl = Chem.SDMolSupplier(sdf, removeHs=False)
    mol_dict = {}
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        if indicies is not None and i not in indicies:
            continue
        mol_dict[i] = mol

    return mol_dict


def batch_calculations(input_file):
    """
    Returns the indicies of molecules first sorted by number of atoms (ascending), then by number of molecules of the same name (descending).
    Args:
        input_file: The path to the input file.
    Returns:
        A dictionary where the keys are the number of atoms in the molecules,
        and the values are dictionaries with molecule names as keys and lists of indexes as values.
    """

    print("Beginning calculations for file processing...", flush=True)

    file_type = input_file.split(".")[-1].strip()

    if file_type == "sdf":
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
    elif file_type == "csv":
        names, smiles = read_csv(input_file)
        suppl = []
        for name, smile in zip(names, smiles):
            if smile is not None:
                mol = Chem.AddHs(Chem.MolFromSmiles(smile))
                mol.SetProp("_Name", name)
                suppl.append(mol)

    # keys are the molecule names,
    # values are lists of index in path and number of atoms

    num_mols = 0

    molecule_dict = {}
    for idx, mol in enumerate(suppl):
        if mol is not None:
            mol = Chem.AddHs(mol)
            num_mols += 1

            num_atoms = mol.GetNumAtoms()

            if num_atoms not in molecule_dict:
                molecule_dict[num_atoms] = {}

            # Want molecules of the same name to stay together
            identifier = mol.GetProp("_Name")

            if identifier not in molecule_dict[num_atoms]:
                molecule_dict[num_atoms][identifier] = []

            molecule_dict[num_atoms][identifier].append(idx)

    print(f"Input file has {num_mols} molecules.", flush=True)

    print("Sorting calculated file information...", flush=True)
    for num_atoms, identifiers in molecule_dict.items():
        # sorts by the number of molecules of the same name (descending)
        sorted_identifiers = {
            identifier: indices
            for identifier, indices in sorted(
                identifiers.items(), key=lambda item: len(item[1]), reverse=True
            )
        }

        molecule_dict[num_atoms] = sorted_identifiers

    # sorts by the number of atoms (ascending)
    return {key: molecule_dict[key] for key in sorted(molecule_dict)}


def batch_files(input_file, int_dir, basename, input_format, batch_info):
    """
    Batches input_file into smaller files based on the number of atoms in each molecule.
    Args:
        input_file: The path to the input file.
        int_dir: The path to the intermediate directory.
        basename: The base name of the input file.
        input_format: The format of the input file (e.g., "sdf", "csv").
        batch_info: The output of batch_calculations function.
    Returns:
        A list of tuples, where each tuple contains the path to a batched file and the total number of atoms in that file.
        Updated batch_info with new indices for each batched file.
    """

    print("Batching files...", flush=True)

    batched_files = []
    updated_batch_info = {}

    if input_format == "csv":
        names, smiles = read_csv(input_file)

        for num_atoms, identifiers in batch_info.items():

            updated_batch_info[num_atoms] = {}

            # number of atoms of molecules of this size
            total_atoms = 0
            new_basename = basename + "_size_" + str(num_atoms) + f".{input_format}"
            new_file = os.path.join(int_dir, new_basename)

            # in reference to the new batched file
            new_index = 0

            with open(new_file, "w") as f:
                f.write("Name,SMILES\n")

                for identifier, index_list in identifiers.items():
                    updated_batch_info[num_atoms][identifier] = []

                    total_atoms += len(index_list) * num_atoms
                    for idx in index_list:
                        updated_batch_info[num_atoms][identifier].append(new_index)
                        new_index += 1
                        f.write(f"{names[idx]},{smiles[idx]}\n")

            batched_files.append((new_file, total_atoms))

    elif input_format == "sdf":

        for num_atoms, identifiers in batch_info.items():

            updated_batch_info[num_atoms] = {}

            # number of atoms of molecules of this size
            total_atoms = 0
            new_basename = basename + "_size_" + str(num_atoms) + f".{input_format}"
            new_file = os.path.join(int_dir, new_basename)

            # indexes are now relative to the new file (based on number of atoms)
            new_index = 0

            full_indicies = []
            for index_list in identifiers.values():
                full_indicies.extend(index_list)

            mol_dict = sdf2mols(input_file, full_indicies)

            with open(new_file, "w") as f:
                writer = Chem.SDWriter(f)
                for identifier, index_list in identifiers.items():

                    updated_batch_info[num_atoms][identifier] = []

                    total_atoms += len(index_list) * num_atoms

                    for idx in index_list:

                        updated_batch_info[num_atoms][identifier].append(new_index)
                        new_index += 1

                        writer.write(mol_dict[idx])

                writer.close()

            batched_files.append((new_file, total_atoms))

    return batched_files, updated_batch_info


def save_chunks(
    input_file, intermediate_dir, configured_memory_gb, do_batch, batch_info
):
    r"""
    Given an input file, divide the file into chunks based on the available memory and the number of jobs.
    If do_batch is True, batch the input file based on number of atoms first and then divide each batch into chunks.
    Args:
        input_file: The path to the input file.
        intermediate_dir: The path to the intermediate directory.
        configured_memory_gb: The amount of memory (in GB) available for the job.
        do_batch: Whether to batch the input file based on number of atoms.
        batch_info: The output of batch_calculations function.
    Returns:
        A tuple of (list of paths to the chunked files, the actual intermediate directory used).
    """

    print("Dividing input file into chunks...", flush=True)

    # Create an intermediate folder.
    basename = os.path.basename(input_file).split(".")[0].strip()
    input_format = input_file.split(".")[-1].strip()
    
    int_dir = intermediate_dir

    # Handle case where directory already exists
    if os.path.exists(int_dir):
        # Find all existing numbered directories
        parent_dir = os.path.dirname(int_dir)
        base_name = os.path.basename(int_dir)
        existing_dirs = [
            d
            for d in os.listdir(parent_dir)
            if d.startswith(base_name) and os.path.isdir(os.path.join(parent_dir, d))
        ]

        # Extract numbers from existing directories
        numbers = [0]  # Start with 0 in case no numbered directories exist
        for d in existing_dirs:
            try:
                num = int(d.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue

        # Use the next available number
        int_dir = f"{int_dir}_{max(numbers) + 1}"

    os.mkdir(int_dir)

    # batching is redundant if there is only one size of molecule
    if do_batch and len(batch_info) > 1:
        files_to_chunk, batch_info = batch_files(
            input_file, int_dir, basename, input_format, batch_info
        )
    else:
        total_atoms = sum(
            num_atoms * sum(len(index_list) for index_list in identifiers.values())
            for num_atoms, identifiers in batch_info.items()
        )
        files_to_chunk = [(input_file, total_atoms)]

        # Consolidate batch_info into a dictionary which does not account for num_atoms per molecule
        batch_info_dict = {}
        for identifiers in batch_info.values():
            for key, value in identifiers.items():
                if key in batch_info_dict:
                    # Combine values to avoid overwriting
                    if isinstance(batch_info_dict[key], list):
                        batch_info_dict[key].extend(value)
                else:
                    batch_info_dict[key] = value

        batch_info = {1: batch_info_dict}  # Wrap in a dict with key 1 for compatibility

        print(f"Total number of atoms in the input file: {total_atoms}", flush=True)

    chunked_files = []

    # Start chunking process
    if input_format == "csv":

        for i, names_dict in enumerate(batch_info.values()):
            batched_file, total_atoms = files_to_chunk[i]

            names, smiles = read_csv(batched_file)
            data_size = len(smiles)

            if total_atoms < 2**20:
                basename = os.path.basename(batched_file).split(".")[0].strip()
                new_basename = basename + "_chunk_" + str(i + 1) + f".{input_format}"
                new_name = os.path.join(int_dir, new_basename)
                chunked_files.append(new_name)  # total files
                with open(new_name, "a") as f:
                    f.write("Name,SMILES\n")
                with open(new_name, "a", newline="") as f:
                    writer = csv.writer(f)
                    for idx in range(data_size):
                        writer.writerow([names[idx], smiles[idx]])
                continue

            # So that we may evenly divide the file into chunks
            num_chunks = math.ceil(total_atoms / 2**20)
            mols_per_chunk = math.ceil(data_size / num_chunks)

            chunks_of_same_size = []

            for i in range(num_chunks):
                basename = os.path.basename(batched_file).split(".")[0].strip()
                new_basename = basename + "_chunk_" + str(i + 1) + f".{input_format}"
                new_name = os.path.join(int_dir, new_basename)
                chunks_of_same_size.append(new_name)  # files of same size
                chunked_files.append(new_name)  # total files
                with open(new_name, "a") as f:
                    f.write("Name,SMILES\n")

            current_chunk = 0
            mols_in_current_chunk = 0

            # Greedily fill each chunk
            for index_list in names_dict.values():
                if (
                    len(index_list) + mols_in_current_chunk > mols_per_chunk
                    and current_chunk < num_chunks - 1
                ):
                    print(
                        f"Job{current_chunk+1}, number of inputs: {mols_in_current_chunk}",
                        flush=True,
                    )
                    mols_in_current_chunk = 0
                    current_chunk += 1

                with open(chunks_of_same_size[current_chunk], "a", newline="") as f:
                    for idx in index_list:
                        writer = csv.writer(f)
                        writer.writerow([names[idx], smiles[idx]])
                        mols_in_current_chunk += 1

            print(
                f"Job{current_chunk+1}, number of inputs: {mols_in_current_chunk}",
                flush=True,
            )

    elif input_format == "sdf":

        for i, names_dict in enumerate(batch_info.values()):
            batched_file, total_atoms = files_to_chunk[i]

            mol_dict = sdf2mols(batched_file)
            data_size = len(mol_dict)

            if total_atoms < 2**20:

                basename = os.path.basename(batched_file).split(".")[0].strip()
                new_basename = basename + "_chunk_" + str(i + 1) + f".{input_format}"
                new_name = os.path.join(int_dir, new_basename)
                chunked_files.append(new_name)  # total files

                with open(new_name, "a") as f:
                    writer = Chem.SDWriter(f)
                    for idx in range(data_size):
                        writer.write(mol_dict[idx])
                    writer.close()
                continue

            num_chunks = math.ceil(total_atoms / 2**20)
            mols_per_chunk = math.ceil(data_size / num_chunks)

            chunks_of_same_size = []

            for i in range(num_chunks):
                basename = os.path.basename(batched_file).split(".")[0].strip()
                new_basename = basename + "_chunk_" + str(i + 1) + f".{input_format}"
                new_name = os.path.join(int_dir, new_basename)
                chunks_of_same_size.append(new_name)  # files of same size
                chunked_files.append(new_name)  # total files

            current_chunk = 0
            mols_in_current_chunk = 0

            for index_list in names_dict.values():
                if (
                    len(index_list) + mols_in_current_chunk > mols_per_chunk
                    and current_chunk < num_chunks - 1
                ):
                    print(
                        f"Job{current_chunk+1}, number of inputs: {mols_in_current_chunk}",
                        flush=True,
                    )
                    mols_in_current_chunk = 0
                    current_chunk += 1

                with open(chunks_of_same_size[current_chunk], "a") as f:
                    writer = Chem.SDWriter(f)
                    for idx in index_list:
                        writer.write(mol_dict[idx])
                        mols_in_current_chunk += 1
                    writer.close()

            print(
                f"Job{current_chunk+1}, number of inputs: {mols_in_current_chunk}",
                flush=True,
            )

    print(
        f"The available memory is {configured_memory_gb} GB. The task will be divided into {len(chunked_files)} job(s).",
        flush=True,
    )

    return chunked_files, int_dir


def combine_files(files, input_file, output_dir, output_suffix="_out.sdf", output_file = None):
    """
    Combine multiple files into a single file.
    Args:
        files: A list of paths to the files to combine.
        input_file: The path to the input file.
        output_dir: The path to the output directory.
        output_suffix: The suffix of the output file.
        output_file: The path to the output file. Will override output_dir if provided.
    Returns:
        The path to the combined file.
    """
    if len(files) == 0:
        msg = """The optimization engine did not run, or no 3D structure converged.
                 The reason might be one of the following:
                 1. Allocated memory is not enough;
                 2. The input SMILES encodes invalid chemical structures;
                 3. Patience is too small."""
        sys.exit(msg)

    suffix, output_type = output_suffix.split(".")
    dot_output_type = "." + output_type
    basename = os.path.basename(input_file).split(".")[0]
    unique_filename = _get_unique_filename(
        output_dir, f"{basename}{suffix}", dot_output_type
    )
    if output_file is None:
        output_path = os.path.join(output_dir, unique_filename)
    else:
        output_path = output_file

    if output_type == "csv":
        data = []
        for file in files:
            with open(file, "r") as f:
                data_i = f.readlines()
            data += data_i

        with open(output_path, "a") as f:

            if input_file.endswith(".csv") and unique_filename.endswith(".csv"):
                f.write("Name,SMILES\n")

            for line in data:
                if ("Name,SMILES") not in line:
                    f.write(line)

    elif output_type == "sdf":
        # This avoids loading all the files into memory at once
        with open(output_path, "a") as f:
            writer = Chem.SDWriter(f)
            for file in files:
                suppl = Chem.SDMolSupplier(file, removeHs=False)
                for mol in suppl:
                    if mol is not None:
                        writer.write(mol)
            writer.close()

    return output_path


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


def _get_unique_filename(output_dir: str, base_filename: str, ext: str) -> str:
    """If a file named base_filename exists, add a number after."""
    n = 1
    unique_filename = f"{base_filename}{ext}"
    while os.path.exists(f"{os.path.join(output_dir, unique_filename)}"):
        unique_filename = base_filename + f"_{n}{ext}"
        n += 1

    return unique_filename

def remove_intermediates(intermediate_dir, input_file: str = None, output_file: str = None, output_dir: str = None):
    r"""
    Remove intermediate files ensuring that the input and output are not inside the
    intermediate directory. If any provided paths are inside the intermediate directory,
    an error is raised to prevent accidental deletion.
    """
    try:
        abs_intdir = os.path.abspath(intermediate_dir) if intermediate_dir else None
        if abs_intdir and os.path.exists(abs_intdir):
            prefix = abs_intdir + os.sep
            protected = []
            if input_file:
                protected.append(os.path.abspath(input_file))
            if output_file:
                protected.append(os.path.abspath(output_file))
            if output_dir:
                protected.append(os.path.abspath(output_dir))

            for p in protected:
                if p == abs_intdir or p.startswith(prefix):
                    raise ValueError(
                        f"{p} must not be inside the intermediate directory when remove_intermediates=True"
                    )

            shutil.rmtree(abs_intdir)
    except Exception as e:
        print(f"Error removing intermediate directory: {e}", flush=True)

    print("Cleanup complete.", flush=True)
    
    