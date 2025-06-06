import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from functools import partial
from typing import Callable, Optional

import ase
import ase.calculators.calculator
import torch
import torchani
from ase import Atoms
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations, VibrationsData
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm.auto import tqdm

from ..batch_opt.ANI2xt_no_rep import ANI2xt
from ..batch_opt.batchopt import EnForce_ANI
from ..model_validation import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1 / hartree2ev


class Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator interface for AIMNET and ANI2xt"""

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)
        self.species = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "As": 33,
            "Se": 34,
            "Br": 35,
            "I": 53,
            "B": 5,
        }

    def set_charge(self, charge: int):
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        species = torch.tensor(
            [self.species[symbol] for symbol in self.atoms.get_chemical_symbols()],
            dtype=torch.long,
            device=self.device,
        )
        coordinates = (
            torch.tensor(self.atoms.get_positions()).to(self.device).to(self.dtype)
        )
        coordinates = coordinates.requires_grad_(True)

        species = species.unsqueeze(0)
        coordinates = coordinates.unsqueeze(0)

        energy, forces = self.model(coordinates, species, self.charge)
        self.results["energy"] = energy.item()
        self.results["forces"] = forces.squeeze(0).to("cpu").numpy()


def get_mol_idx_t1(mol):
    """Get idx and temperature from openbabel molecule"""

    idx = str(mol).split()[1].strip().split("/")[-1].strip().split(".")[0]
    T = int(idx.split("-")[-1])
    return (idx, T)


def get_mol_idx_t3(mol):
    "Setting default index and temperature"
    idx = ""
    T = 298
    return (idx, T)


def mol2aimnet_input(mol: Chem.Mol, device=torch.device("cpu")) -> dict:
    """Converts mol to aimnet input, assuming the mol has only 1 conformer."""
    conf = mol.GetConformer()
    coord = torch.tensor(conf.GetPositions(), device=device).unsqueeze(0)
    numbers = torch.tensor(
        [a.GetAtomicNum() for a in mol.GetAtoms()], device=device
    ).unsqueeze(0)
    charge = torch.tensor([Chem.GetFormalCharge(mol)], device=device, dtype=torch.float)
    return dict(coord=coord, numbers=numbers, charge=charge)


def model_name2model_calculator(model_name: str, device=torch.device("cpu"), charge=0):
    """Return a model and the ASE calculator object for a molecule"""
    if model_name == "ANI2xt":
        model = EnForce_ANI(
            ANI2xt(device, periodic_table_index=True), model_name
        ).double()
        calculator = Calculator(model, charge)
    elif model_name == "AIMNET":
        # Using the ensemble AIMNet2 model for computing energy and forces
        aimnet = torch.jit.load(
            os.path.join(root, "models/aimnet2_wb97m_ens_f.jpt"), map_location=device
        )
        model = EnForce_ANI(aimnet, model_name)
        calculator = Calculator(model, charge)
    elif model_name == "AIMNET-lite":
        ## TO BE IMPLEMENTED
        aimnet = torch.jit.load(
            os.path.join(root, "models/aimnet2_wb97m-d3_0.jpt"), map_location=device
        )
        model = EnForce_ANI(aimnet, model_name)
        calculator = Calculator(model, charge)
    elif model_name == "ANI2x":
        ani2x = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
        model = EnForce_ANI(ani2x, model_name)
        # calculator = ani2x.ase()
        calculator = Calculator(model, charge)
    elif os.path.exists(model_name):
        user_nnp = torch.jit.load(model_name, map_location=device).double()
        model = EnForce_ANI(user_nnp, model_name)
        calculator = Calculator(model, charge)
    else:
        raise ValueError(
            "model has to be 'ANI2x', 'ANI2xt', 'AIMNET' or a path to a userNNP model."
        )
    return model, calculator


def mol2atoms(mol: Chem.Mol):
    """convert a RDKit mol object to ASE atoms object"""
    coord = mol.GetConformer().GetPositions()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    atoms = Atoms(species, coord)
    return atoms


def vib_hessian(
    mol: Chem.Mol,
    ase_calculator,
    model,
    device=torch.device("cpu"),
    model_name="AIMNET",
):
    """return a VibrationsData object
    model: ANI2xt or AIMNet2 or ANI2x or userNNP that can be used to calculate Hessian
    """
    # get the ASE atoms object
    coord = mol.GetConformer().GetPositions()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    charge = rdmolops.GetFormalCharge(mol)
    atoms = Atoms(species, coord)
    atoms.calc = ase_calculator

    # get the Hessian
    coord = torch.tensor(coord).to(device).unsqueeze(0)
    num_atoms = coord.shape[1]
    numbers = torch.tensor([[a.GetAtomicNum() for a in mol.GetAtoms()]]).to(device)
    charge = torch.tensor(charge).to(device)

    hess_helper = partial(
        aimnet_hessian_helper,
        numbers=numbers,
        charge=charge,
        model=model,
        model_name=model_name,
    )
    hess = torch.autograd.functional.hessian(hess_helper, coord)
    hess = hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()

    # get the VibrationsData object
    vib = VibrationsData(atoms, hess)
    return vib


def vib_ase(mol: Chem.Mol, ase_calculator):
    """return a VibrationsData object
    model: ANI2xt or AIMNet2 model with EnForce_ANI wrapper"""
    # get the ASE atoms object
    atoms = mol2atoms(mol)
    atoms.calc = ase_calculator

    # get the VibrationsData object
    vib = Vibrations(atoms)
    vib.clean()
    vib.run()
    return vib


def do_mol_thermo(
    mol: Chem.Mol,
    atoms: ase.Atoms,
    model: torch.nn.Module,
    device=torch.device("cpu"),
    T=298.0,
    model_name="AIMNET-lite",
):
    """For a RDKit mol object, calculate its thermochemistry properties.
    model: ANI2xt or AIMNet2 or ANI2x or userNNP that can be used to calculate Hessian
    """
    vib = vib_hessian(mol, atoms.calc, model, device, model_name=model_name)
    vib_e = vib.get_energies()
    e = atoms.get_potential_energy()
    thermo = IdealGasThermo(
        vib_energies=vib_e,
        potentialenergy=e,
        atoms=atoms,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0,
        ignore_imag_modes=True,
    )
    H = thermo.get_enthalpy(temperature=T) * ev2hatree
    S = thermo.get_entropy(temperature=T, pressure=101325) * ev2hatree
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325) * ev2hatree

    mol.SetProp("H_hartree", str(H))
    mol.SetProp("S_hartree", str(S))
    mol.SetProp("T_K", str(T))
    mol.SetProp("G_hartree", str(G))
    mol.SetProp("E_hartree", str(e * ev2hatree))

    # Updating ASE atoms coordinates into mol
    coord = atoms.get_positions()
    for i, atom in enumerate(mol.GetAtoms()):
        mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
    return mol


def aimnet_hessian_helper(
    coord: torch.tensor,
    numbers: Optional[torch.Tensor] = None,
    charge: Optional[torch.Tensor] = None,
    model: Optional[torch.nn.Module] = None,
    model_name="AIMNET-lite",
):
    """coord shape: (1, num_atoms, 3)
    numbers shape: (1, num_atoms)
    charge shape: (1,)"""
    if model_name == "AIMNET-lite":
        dct = dict(coord=coord, numbers=numbers, charge=charge)
        return model(dct)["energy"]  # energy unit: eV
    elif model_name == "ANI2xt":
        device = coord.device
        periodict2idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
        numbers2 = torch.tensor(
            [periodict2idx[num.item()] for num in numbers.squeeze()], device=device
        ).unsqueeze(0)
        e = model(numbers2, coord)
        return e  # energy unit: eV
    elif model_name == "ANI2x":
        e = model((numbers, coord)).energies * hartree2ev
        return e  # energy unit: eV
    elif os.path.exists(model_name):
        e = model.forward(numbers, coord, charge)
        return e  # energy unit: eV


def calc_thermo(
    model_name: str,
    input_file: str,
    output_file: Optional[str] = None,
    mol_info_func: Optional[Callable] = None,
    use_gpu: bool = True,
    gpu_idx: int = 0,
    opt_tol: float = 0.0002,
    opt_steps: int = 5000,
):
    """
    ASE interface for calculation thermo properties using ANI2x, ANI2xt or AIMNET.

    :param path: Input sdf file
    :type path: str
    :param model_name: ANI2x, ANI2xt, AIMNET or a path to a userNNP model
    :type model_name: str
    :param mol_info_func: A function that returns the name and temperature (idx, T)
                          from a rdkit mol object. If not provided, the
                          thermodynamic properties will be calculated at 298 K
    :type mol_info_func: function, optional
    :param gpu_idx: GPU cuda index, defaults to 0
    :type gpu_idx: int, optional
    :param opt_tol: Convergence_threshold for geometry optimization, defaults to 0.0002
    :type opt_tol: float, optional
    :param opt_steps: Maximum geometry optimization steps, defaults to 5000
    :type opt_steps: int, optional
    """
    # Prepare output name
    if output_file is None:
        dir = os.path.dirname(input_file)
        if os.path.exists(model_name):
            basename = os.path.basename(input_file).split(".")[0] + "_userNNP_G.sdf"
        else:
            basename = (
                os.path.basename(input_file).split(".")[0] + f"_{model_name}_G.sdf"
            )
        output_file = os.path.join(dir, basename)
    writer = Chem.SDWriter(output_file)

    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    if model_name == "AIMNET-lite":
        aimnet0_path = os.path.join(root, "models/aimnet2_wb97m-d3_0.jpt")
        hessian_model = torch.jit.load(aimnet0_path, map_location=device)
    elif model_name == "ANI2xt":
        hessian_model = ANI2xt(device).double()
    elif model_name == "ANI2x":
        hessian_model = (
            torchani.models.ANI2x(periodic_table_index=True).to(device).double()
        )
    elif os.path.exists(model_name):
        hessian_model = torch.jit.load(model_name, map_location=device).double()
    model, calculator = model_name2model_calculator(model_name, device)

    out_mols, mols_failed = [], []
    for mol in tqdm(Chem.SDMolSupplier(input_file, removeHs=False)):
        coord = mol.GetConformer().GetPositions()
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        charge = rdmolops.GetFormalCharge(mol)
        atoms = Atoms(species, coord)

        calculator.set_charge(charge)
        atoms.calc = calculator

        if mol_info_func is None:
            idx = mol.GetProp("_Name").strip()
            T = 298
        else:
            idx, T = mol_info_func(mol)

        try:
            try:
                try:
                    EnForce_in = mol2aimnet_input(mol, device)
                    _, f_ = model(
                        EnForce_in["coord"].requires_grad_(True),
                        EnForce_in["numbers"],
                        EnForce_in["charge"],
                    )
                    fmax = f_.norm(dim=-1).max(dim=-1)[0].item()
                    assert fmax <= 0.01, "fmax too large"
                    mol = do_mol_thermo(
                        mol, atoms, hessian_model, device, T, model_name=model_name
                    )
                    out_mols.append(mol)
                except AssertionError:
                    print("Optimizing the input geometry...", flush=True)
                    opt = BFGS(atoms)
                    opt.run(fmax=3e-3, steps=opt_steps)
                    mol = do_mol_thermo(
                        mol, atoms, hessian_model, device, T, model_name=model_name
                    )
                    out_mols.append(mol)
            except ValueError:
                print(
                    "Tighter convergence threshold for geometry optimization",
                    flush=True,
                )
                opt = BFGS(atoms)
                opt.run(fmax=opt_tol, steps=opt_steps)
                mol = do_mol_thermo(
                    mol, atoms, hessian_model, device, T, model_name=model_name
                )
                out_mols.append(mol)
        except:
            print("Failed: ", idx, flush=True)
            mols_failed.append(mol)
        writer.write(mol)
    writer.close()

    print("Number of failed thermo calculations: ", len(mols_failed), flush=True)
    print("Number of successful thermo calculations: ", len(out_mols), flush=True)

    return output_file
