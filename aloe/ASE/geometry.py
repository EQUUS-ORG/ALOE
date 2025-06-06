import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import torch
from rdkit import Chem

from ..batch_opt.batchopt import optimizing
from ..model_validation import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def opt_geometry(path: str, model_name: str, gpu_idx=0, opt_tol=0.003, opt_steps=5000):
    """
    Geometry optimization interface with FIRE optimizer.

    :param path: Input sdf file
    :type path: str
    :param model_name: ANI2x, ANI2xt, AIMNET or a path to userNNP model
    :type model_name: str
    :param gpu_idx: GPU cuda index, defaults to 0
    :type gpu_idx: int, optional
    :param opt_tol: Convergence_threshold for geometry optimization (eV/A), defaults to 0.003
    :type opt_tol: float, optional
    :param opt_steps: Maximum geometry optimization steps, defaults to 5000
    :type opt_steps: int, optional
    """
    ev2hatree = 1 / hartree2ev
    # create output path that is in the same directory as the input file
    dir = os.path.dirname(path)
    if os.path.exists(path):
        basename = os.path.basename(path).split(".")[0] + f"_userNNP_opt.sdf"
    else:
        basename = os.path.basename(path).split(".")[0] + f"_{model_name}_opt.sdf"
    outpath = os.path.join(dir, basename)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    opt_config = {
        "opt_steps": opt_steps,
        "opttol": opt_tol,
        "patience": opt_steps,
        "batchsize_atoms": 1024,
    }
    opt_engine = optimizing(path, outpath, model_name, device, opt_config)
    opt_engine.run()

    # change the energy unit from ev to hartree
    mols = list(Chem.SDMolSupplier(outpath, removeHs=False))
    with Chem.SDWriter(outpath) as f:
        for mol in mols:
            e = float(mol.GetProp("E_tot")) * ev2hatree
            mol.SetProp("E_tot", str(e))
            f.write(mol)
    return outpath
