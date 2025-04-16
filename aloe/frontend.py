import asyncio
import functools
import os
from dataclasses import asdict, dataclass
from typing import List, Union

from aloe.backend import (
    calculate_thermo,
    embed_conformers,
    generate_stereoisomers,
    optimize_conformers,
    rank_conformers,
)
from aloe.bdfe_calculation.bdfe_calc import (
    bdfe_calculator,
    products_pipeline,
    write_failed_reactants,
)
from aloe.bdfe_calculation.product_generator import generate_products
from aloe.file_utils import (
    batch_calculations,
    combine_files,
    make_output_name,
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
    k: int = None
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
        self.hardware_settings = self.hardware_settings["gpu_idx"]

        # Ensure all required parameters are set before running the pipeline
        return chunks

    async def run(self):
        r"""
        This function runs the aloe pipeline. Choose which functions to run and optionally set parameters for each function.
        Returns:
            str, path to the output file.

        """

        chunks = self.prepwork()

        output_files = await run_auto3D_pipeline(
            chunks,
            self.selected_functions,
            self.user_parameters,
            self.hardware_settings,
        )

        if (
            len(self.selected_functions) == 1
            and self.selected_functions[0] == "gen_isomer"
        ):
            output_suffix = "_out.csv"
        else:
            output_suffix = "_out.sdf"

        if self.output_dir is None:
            self.output_dir = os.path.dirname(self.input_file)

        return combine_files(
            output_files, self.input_file, self.output_dir, output_suffix
        )

    def calculate_bdfe(self):
        r"""
        Generates the products from a list of reactants and calulates the change in bond dissociation free energy (BDFE) for each reaction.

        Returns:
            output_file str: path to the output file containing the BDFE calculations.
            failed_file str: path to the file containing the failed reactants (if any).
        """

        reactant_chunks, hardware_settings = self.prepwork()
        product_chunks = []
        failed_reactants = []

        for chunk in reactant_chunks:
            product_chunk, failed = products_pipeline(chunk)
            product_chunks.append(product_chunk)
            failed_reactants.extend(failed)

        # run pipeline on both files

        self.add_step(ConformerConfig())
        self.add_step(OptConfig())
        self.add_step(RankConfig(k=1))
        self.add_step(ThermoConfig())

        reactant_files = asyncio.run(
            run_auto3D_pipeline(
                reactant_chunks,
                self.selected_functions,
                self.user_parameters,
                hardware_settings,
            )
        )

        product_files = asyncio.run(
            run_auto3D_pipeline(
                product_chunks,
                self.selected_functions,
                self.user_parameters,
                hardware_settings,
            )
        )

        # Calculates BDFEs
        output_files = []
        for reactant, product in zip(reactant_files, product_files):
            output_files.append(bdfe_calculator(reactant, product))

        if self.output_dir is None:
            self.output_dir = os.path.dirname(self.input_file)

        output_file = combine_files(
            output_files, self.input_file, self.output_dir, "_bdfe.csv"
        )

        failed_file = write_failed_reactants(output_file, failed_reactants)

        return output_file, failed_file


async def run_gen(input_file, **kwargs):
    """Generate isomers async wrapper"""
    return generate_stereoisomers(input_file, **kwargs)


async def run_embed(input_file, **kwargs):
    """Embed conformers async wrapper"""
    return embed_conformers(input_file, **kwargs)


async def run_opt(input_file, **kwargs):
    """Optimize conformers async wrapper"""
    return optimize_conformers(input_file, **kwargs)


async def run_rank(input_file, **kwargs):
    """Rank conformers async wrapper"""
    return rank_conformers(input_file, **kwargs)


async def run_thermo(input_file, **kwargs):
    """Calculate thermochemistry async wrapper"""
    return calculate_thermo(input_file, **kwargs)


FUNCTIONS = {
    "StereoIsoConfig": run_gen,
    "ConformerConfig": run_embed,
    "OptConfig": run_opt,
    "RankConfig": run_rank,
    "ThermoConfig": run_thermo,
}


async def process_chunk(chunk, pipeline, gpu_index):
    r"""
    Process a chunk of data through the pipeline.
    Args:
        chunk: str, path to the input file chunk.
        pipeline: list of functions to run on the chunk.
        gpu_index: int, GPU index to use for processing.
    Returns:
        chunk: str, path to the output file.
    """

    for partial_func in pipeline:
        if (
            partial_func.func.__name__ == "run_opt"
            or partial_func.func.__name__ == "run_thermo"
        ):
            partial_func = functools.partial(partial_func, gpu_idx=gpu_index)
        try:
            chunk = await partial_func(input_file=chunk)
        except Exception as e:
            print("ERROR: ", e, flush=True)

    return chunk


async def worker(queue, pipeline, gpu_index, output_queue):
    r"""
    Assigns chunk to one gpu_index
    Args:
        queue: asyncio.Queue, queue of chunks to process.
        pipeline: list of functions to run on the chunk.
        gpu_index: int, GPU index to use for processing.
        output_queue: asyncio.Queue, queue to store the results.
    """
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        res = await process_chunk(chunk, pipeline, gpu_index)
        queue.task_done()
        output_queue.put_nowait(res)


async def get_list_from_queue(queue):
    r"""
    Get all items from the queue.
    Args:
        queue: asyncio.Queue, queue to get items from.
    """
    items = []
    while not queue.empty():
        item = await queue.get()
        items.append(item)
        queue.task_done()  # Mark the item as processed
    return items


async def run_auto3D_pipeline(chunks, selected_functions, parameters, gpu_indicies):
    r"""
    Run the auto3D pipeline asynchronously.
    Args:
        chunks: list of str, paths to the input file chunks.
        selected_functions: list of str, functions to run on the chunks.
        parameters: dict, parameters for each function.
        gpu_indicies: list of int, GPU indices to use for processing.
    Returns:
        list of str, paths to the output files.
    """

    pipeline = [
        functools.partial(FUNCTIONS[func], **parameters[func])
        for func in selected_functions
    ]

    queue = asyncio.Queue()
    for chunk in chunks:
        await queue.put(chunk)

    output_queue = asyncio.Queue()

    tasks = []
    for gpu_index in gpu_indicies:
        task = asyncio.create_task(worker(queue, pipeline, gpu_index, output_queue))
        tasks.append(task)

    await queue.join()

    for _ in gpu_indicies:
        await queue.put(None)

    await asyncio.gather(*tasks)

    return await get_list_from_queue(output_queue)
