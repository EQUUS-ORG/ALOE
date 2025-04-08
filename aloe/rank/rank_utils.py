from typing import Iterable, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign


def check_connectivity(mol: Chem.Mol) -> bool:
    r"""
    Check if the generated conformer has the same topology as the original molecule.
    https://github.com/jensengroup/xyz2mol
    """

    block = Chem.MolToXYZBlock(mol, confId=0)
    new_mol = Chem.MolFromXYZBlock(block)
    rdDetermineBonds.DetermineConnectivity(new_mol)
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    adjacency_matrix_new = Chem.GetAdjacencyMatrix(new_mol)
    return np.allclose(adjacency_matrix, adjacency_matrix_new, atol=1e-1)


def are_equal(mol_i: Chem.Mol, mol_j: Chem.Mol, threshold: float) -> bool:
    r"""
    Check whether two conformers are the same.
    Caution: This function does not verify whether the two molecules have the same topology!
    """
    # temperoray bug fix for https://github.com/rdkit/rdkit/issues/6826
    # removing Hs speeds up the calculation
    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol_i), Chem.RemoveHs(mol_j))
    return rmsd < threshold


def filter_unique(
    mols: Iterable[Chem.Mol], threshold: float = 0.3, k: int = -1
) -> List[Chem.Mol]:
    """
    Remove structures that are very similar.

    Arguments:
        mols: list of rdkit mol objects
        threshold: maximum RMSD for two molecules to be considered the same
        k: only consider the first k unique molecules in the list
    Returns:
        unique_mols: unique molecules
    """

    # Remove similar structures
    unique_mols = []
    for mol_i in filter(None, mols):
        unique = True
        for mol_j in unique_mols:
            try:
                unique = not are_equal(mol_i=mol_i, mol_j=mol_j, threshold=threshold)
            except RuntimeError:
                unique = False
            if not unique:
                break
        if unique:
            unique_mols.append(mol_i)
        if k > 0 and len(unique_mols) >= k:
            break
    return unique_mols
