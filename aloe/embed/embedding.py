import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


def embed_conformer(smi: str, max_conformers, np, threshold) -> Chem.Mol:
    """Embed conformers for a smi"""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    if max_conformers is None:
        # Determined using the formula proposed in: https://doi.org/10.1021/acs.jctc.0c01213
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        n_conformers = min(
            max(num_heavy_atoms, int(2 * 8.481 * (num_rotatable_bonds**1.642))),
            1000,
        )
    else:
        n_conformers = max_conformers
    try:
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_conformers,
            randomSeed=42,
            numThreads=np,
            pruneRmsThresh=threshold,
            maxAttempts=10,  # https://github.com/rdkit/rdkit/discussions/6804
        )
        return mol
    except Exception as e:
        return None


def min_pairwise_distance(points: np.array) -> float:
    """
    Finds the minimum pairwise distance among the n points provided in a n x 3 matrix.

    Parameters:
    points (numpy.ndarray): A n x 3 matrix representing the coordinates of n points in 3D space.

    Returns:
    float: The minimum pairwise distance among the n points.
    """
    # Ensure input is a NumPy array
    points = points.astype(np.float32)
    n = points.shape[0]
    # Expand dimensions of points to enable broadcasting
    points_expanded = np.expand_dims(points, axis=1).repeat(n, axis=1)

    # Compute pairwise squared differences
    diff_squared = (points_expanded - points_expanded.transpose(1, 0, 2)) ** 2

    # Sum along the last dimension to get pairwise squared distances
    pairwise_squared_distances = np.sum(diff_squared, axis=-1)

    # Find the minimum squared distance
    upp_indices = np.triu_indices(n, 1)
    upp_values = pairwise_squared_distances[upp_indices]
    min_squared_distance = np.min(upp_values)

    # Return the square root of the minimum squared distance
    return np.sqrt(min_squared_distance)
