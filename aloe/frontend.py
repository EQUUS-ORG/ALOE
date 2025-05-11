import functools
import multiprocessing
import os
from dataclasses import asdict, dataclass
from queue import Queue
from typing import List

from aloe.backend import (
    calculate_thermo,
    embed_conformers,
    generate_stereoisomers,
    optimize_conformers,
    rank_conformers,
)
from aloe.file_utils import (
    batch_calculations,
    combine_files,
    save_chunks,
    update_hardware_settings,
)
from aloe.model_validation import check_shared_parameters


@dataclass
class StereoIsoConfig:
    r"""
    enumerate_tautomers: bool, whether to enumerate tautomers, default False
    onlyUnassigned: bool, whether to only generate unassigned stereoisomers, default False
    unique: bool, whether to generate unique stereoisomers, default True
    """
    enumerate_tautomers: bool = False
    onlyUnassigned: bool = True
    unique: bool = True


@dataclass
class ConformerConfig:
    r"""
    max_conformers: int, maximum number of conformers to generate, default None
    mpi_np: int, number of CPU cores for isomer generation, default 4
    threshold: float RMSD threshold for considering conformers as duplications, default 0.3
    """
    max_conformers: int = None
    mpi_np: int = 4
    threshold: float = 0.3


@dataclass
class OptConfig:
    r"""
    Arguemnts:
    optimizing_engine: str, Geometry optimization engine, default "AIMNET-lite"
    patience: int, maximum consecutive steps without force decrease before termination, defaults to 1000.
    opt_steps: int, maximum optimization steps per structure, defaults to 5000.
    convergence_threshold: float, Maximum force threshold for convergence, defaults to 0.003.
    """
    optimizing_engine: str = "AIMNET-lite"
    patience: int = 1000
    opt_steps: int = 5000
    convergence_threshold: float = 0.003


@dataclass
class RankConfig:
    r"""
    k or window must be provided.
    Arguments:
    k: int, number of lowest-energy structures to select, default None
    window: bool, whether to output structures with energies within x kcal/mol from the lowest energy conformer, defaults to None.
    threshold: float, RMSD threshold for considering conformers as duplicates, defaults to 0.3.
    """
    k: int = 1
    window: bool = None
    threshold: float = 0.3


@dataclass
class ThermoConfig:
    r"""
    Arguments:
    model_name: str: name of the forcefield to use, defaults to "AIMNET-lite".
    mol_into_func: Callable, function to convert the molecule into a format that can be used by the forcefield, defaults to None.
    opt_tol: float, Convergence_threshold for geometry optimization, defaults to 0.0002.
    opt_steps: int, Maximum optimization steps per structure, defaults to 5000.
    """
    model_name: str = "AIMNET-lite"
    mol_info_func: callable = None
    opt_tol: float = 0.0002
    opt_steps: int = 5000


class aloe:
    def __init__(self, input_file, output_dir=None, **kwargs):
        r"""
        Arguments:
        input_file: str, path to the input file (.csv if starting with isomer generation or embedding, .sdf otherwise).
        output_file: str, path to the output file, default None
        use_gpu: bool, whether to use GPU for calculations, default False.
        gpu_idx: int or List[int], Only applies when use_gpu=True. GPU index to use for calculations, default 0.
        do_batch: bool, whether to batch calculations based on molecule sizes, default False.
        memory: int, RAM allocation for Auto3D in GB, default None.
        batchsize_atoms: int, Number of atoms per batch for geometry calculations, default 2048.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.selected_functions = []
        self.user_parameters = {
            "StereoIsoConfig": asdict(StereoIsoConfig()),
            "ConformerConfig": asdict(ConformerConfig()),
            "OptConfig": asdict(OptConfig()),
            "RankConfig": asdict(RankConfig()),
            "ThermoConfig": asdict(ThermoConfig()),
        }
        self.hardware_settings = {
            "use_gpu": kwargs.get("use_gpu", False),
            "gpu_idx": kwargs.get("gpu_idx", 0),
            "do_batch": kwargs.get("do_batch", False),
            "memory": kwargs.get("memory", None),
            "batchsize_atoms": kwargs.get("batchsize_atoms", 2048),
        }

    def add_step(self, config):
        func = config.__class__.__name__
        self.selected_functions.append(func)
        self.user_parameters[func] = asdict(config)

    def prepwork(self):
        r"""
        Prepares the aloe pipeline for execution. This function can be used to ensure that all necessary parameters are set before running the pipeline.
        Returns:
            None
        """
        if not self.selected_functions:
            raise ValueError("No steps have been added to the pipeline.")

        check_shared_parameters(
            self.user_parameters["OptConfig"], self.user_parameters["ThermoConfig"]
        )

        # batch info contains a dictionary of key (num_atoms): value (dictionary of key (molecule type): value (number of molecules))
        batch_info = batch_calculations(self.input_file)

        t, self.hardware_settings["batchsize_atoms"] = update_hardware_settings(
            self.hardware_settings
        )

        # Assign hardware settings to respective functions
        self.user_parameters["OptConfig"]["use_gpu"] = self.hardware_settings["use_gpu"]
        self.user_parameters["ThermoConfig"]["use_gpu"] = self.hardware_settings[
            "use_gpu"
        ]
        self.user_parameters["OptConfig"]["batchsize_atoms"] = self.hardware_settings[
            "batchsize_atoms"
        ]

        chunks = save_chunks(
            self.input_file, t, self.hardware_settings["do_batch"], batch_info
        )

        # Consolidate into one list
        if isinstance(self.hardware_settings["gpu_idx"], int):
            self.hardware_settings["gpu_idx"] = [self.hardware_settings["gpu_idx"]]

        # This is the only relevant hardware setting for the pipeline
        # -1 to indicate only CPU calculations
        self.gpu_indices = (
            self.hardware_settings["gpu_idx"]
            if self.hardware_settings["use_gpu"]
            else [-1]
        )
        # Ensure all required parameters are set before running the pipeline
        return chunks

    def run(self):
        r"""
        This function runs the aloe pipeline. Choose which functions to run and optionally set parameters for each function.
        Returns:
            str, path to the output file.

        """

        chunks = self.prepwork()

        output_files = run_auto3D_pipeline(
            chunks=chunks,
            selected_functions=self.selected_functions,
            user_parameters=self.user_parameters,
            gpu_indicies=self.gpu_indices,
        )

        if (
            len(self.selected_functions) == 1
            and self.selected_functions[0] == "StereoIsoConfig"
        ):
            output_suffix = "_out.csv"
        else:
            output_suffix = "_out.sdf"

        if self.output_dir is None:
            self.output_dir = os.path.dirname(self.input_file)

        return combine_files(
            output_files, self.input_file, self.output_dir, output_suffix
        )


def run_gen(input_file, **kwargs):
    """Generate isomers async wrapper"""
    return generate_stereoisomers(input_file, **kwargs)


def run_embed(input_file, **kwargs):
    """Embed conformers async wrapper"""
    return embed_conformers(input_file, **kwargs)


def run_opt(input_file, **kwargs):
    """Optimize conformers async wrapper"""
    return optimize_conformers(input_file, **kwargs)


def run_rank(input_file, **kwargs):
    """Rank conformers async wrapper"""
    return rank_conformers(input_file, **kwargs)


def run_thermo(input_file, **kwargs):
    """Calculate thermochemistry async wrapper"""
    return calculate_thermo(input_file, **kwargs)


FUNCTIONS = {
    "StereoIsoConfig": run_gen,
    "ConformerConfig": run_embed,
    "OptConfig": run_opt,
    "RankConfig": run_rank,
    "ThermoConfig": run_thermo,
}


# === Pipeline ===
def run_pipeline(input_file, pipeline, gpu_index):
    current_file = input_file
    for step in pipeline:
        if step is not None:
            if step.func.__name__ == "run_opt" or step.func.__name__ == "run_thermo":
                step = functools.partial(step, gpu_idx=gpu_index)

            # try:
            current_file = step(input_file=current_file)
            # except Exception as e:
            # print(f"Error processing step {step.func.__name__}: {e}", flush=True)

    return current_file


# === GPU Worker ===
def gpu_worker(gpu_id, input_file, pipeline, result_queue):
    if not gpu_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    final_output = run_pipeline(input_file, pipeline, gpu_id)
    result_queue.put((gpu_id, final_output))


# === Job Manager ===
def gpu_job_manager(jobs, available_gpus):
    result_queue = multiprocessing.Queue()
    gpu_pool = Queue()
    for gpu in available_gpus:
        gpu_pool.put(gpu)

    final_outputs = []
    job_idx = 0
    running = 0

    while job_idx < len(jobs) or running > 0:
        # Dispatch job if there's a free GPU and jobs remain
        if job_idx < len(jobs) and not gpu_pool.empty():
            gpu_id = gpu_pool.get()
            input_file, enabled_steps = jobs[job_idx]
            p = multiprocessing.Process(
                target=gpu_worker,
                args=(gpu_id, input_file, enabled_steps, result_queue),
            )
            p.start()
            job_idx += 1
            running += 1

        # If all GPUs are busy or no jobs left, wait for a result
        elif running > 0:
            gpu_id, final_output = result_queue.get()
            final_outputs.append(final_output)
            gpu_pool.put(gpu_id)
            running -= 1

    return final_outputs


def run_auto3D_pipeline(
    chunks: List[str],
    selected_functions: List[str],
    user_parameters: dict,
    gpu_indicies: List[int],
):
    r"""
    Runs the Auto3D pipeline on the given chunks of data.
    Args:
        chunks: list of str, paths to input files.
        selected_functions: list of str, names of functions to run.
        user_parameters: dict, user-defined parameters for the functions.
        hardware_settings: int or list of int, GPU index or indices to use for processing.
    Returns:
        list of str, paths to output files.
    """
    multiprocessing.set_start_method("spawn", force=True)

    pipeline = [
        functools.partial(FUNCTIONS[func], **user_parameters[func])
        for func in selected_functions
    ]

    jobs = [(chunk, pipeline) for chunk in chunks]

    return gpu_job_manager(jobs, gpu_indicies)
