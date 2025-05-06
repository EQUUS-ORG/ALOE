import csv
from rdkit import Chem

Chem.SetUseLegacyStereoPerception(False)
from rdkit.Chem import Mol
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

from aloe.file_utils import read_csv_dict


class rd_enumerate_isomer(object):
    """
    enumerating stereoisomers starting from an CSV file.

    """

    def __init__(
        self,
        csv: str,
        enumerated_csv: str,
        enumerate_tauts: bool,
        onlyUnassigned: bool,
        unique: bool,
    ):
        """
        csv: the path to the csv file
        enumerated_sdf: the path to the output csv file
        enumerate_tauts: whether to enumerate tautomers
        onlyUnassigned: whether to enumerate only unassigned stereocenters
        unique: whether to enumerate only unique stereoisomers
        """
        self.csv = csv
        self.enumerated_csv = enumerated_csv
        self.enumerate_tauts = enumerate_tauts
        self.onlyUnassigned = onlyUnassigned
        self.unique = unique

    def taut(self):
        """Enumerating tautomers for the input_f csv file"""
        enumerator = rdMolStandardize.TautomerEnumerator()
        data = read_csv_dict(self.csv)
        tautomers = {}
        for key, val in data.items():
            mol = Chem.MolFromSmiles(val)
            tauts = enumerator.Enumerate(mol)
            for i, taut in enumerate(tauts):
                smiles = Chem.MolToSmiles(taut, isomericSmiles=True, doRandom=False)
                tautomers[f"{key}-tautomer{i}"] = smiles
        return tautomers

    def to_isomers(self, mol: Mol) -> list[Mol]:
        r"""
        Args:
            mol (Mol): A molecule.

        Returns:
            list[Mol]: A list of stereoisoemrs.
        """
        options = StereoEnumerationOptions(
            onlyUnassigned=self.onlyUnassigned, unique=self.unique
        )
        isomers = list(EnumerateStereoisomers(mol, options=options))
        return isomers

    def isomer_hack(self, smi: str) -> list[str]:
        r"""
        Generates all possible isomers of a given SMILES string.
        Args:
            smi (str): The SMILES string of the molecule.
        Returns:
            list[str]: A list of SMILES strings representing all possible isomers.
        """

        # first iteration, deal with most common cases
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smi))
        isomers = self.to_isomers(mol)

        # second iteration, deal with imines and other =X-H explicit hydrogens
        second_isomers = []
        for isomer in isomers:
            second_isomers += self.to_isomers(Chem.AddHs(isomer))

        return sorted(
            set(
                [
                    Chem.CanonSmiles(Chem.MolToSmiles(isomer))
                    for isomer in second_isomers
                ]
            )
        )

    def run(self):
        data = read_csv_dict(self.csv)

        if self.enumerate_tauts:
            data = self.taut()

        with open(self.enumerated_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Name", "SMILES"])
            for key, val in data.items():
                isomers = self.isomer_hack(val)

                for i, smi in enumerate(isomers):
                    writer.writerow([f"{key}-isomer{i}", smi])

        return self.enumerated_csv


diimine_pattern = Chem.MolFromSmarts("N=C-C=N")

def EZ_helper(mol, bond_idx):
    """
    Assigns E/Z stereochemistry to the molecule.
    """

    # Create a copy to get stereo atoms for double bonds
    mol_copy = mol.__copy__()
    for id in bond_idx:
        bond = mol_copy.GetBondWithIdx(id)
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOANY)

    # https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/EnumerateStereoisomers.py
    sinfo = Chem.FindPotentialStereo(mol_copy)
    for si in sinfo:
        if si.type == Chem.StereoType.Bond_Double and si.centeredOn in bond_idx:
            bond = mol_copy.GetBondWithIdx(si.centeredOn)
            if not bond.GetStereoAtoms():
                if (
                    si.controllingAtoms[0] == Chem.StereoInfo.NOATOM
                    or si.controllingAtoms[2] == Chem.StereoInfo.NOATOM
                ):
                    continue
                original_bond = mol.GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                )
                original_bond.SetStereoAtoms(
                    si.controllingAtoms[0], si.controllingAtoms[2]
                )
    return mol

def enumerate_EZ_diimine(mol: Mol) -> list[tuple[str, str]]:
    """
    (E, E), (E, Z), (Z,E), (Z, Z)

    Returns:
        list[tuple[str, str]]: A list of tuples of the form (code, SMILES).
    """


    try:
        assert len(mol.GetSubstructMatches(diimine_pattern)) == 1, "More than one diimine found in the molecule"
        match = mol.GetSubstructMatches(diimine_pattern)[0]
        N1, C1 = match[0], match[1]
        N2, C2 = match[-1], match[-2]
        bond1_id = mol.GetBondBetweenAtoms(N1, C1).GetIdx()
        bond2_id = mol.GetBondBetweenAtoms(N2, C2).GetIdx()

        list_of_smiles = []

        # (E, E)
        mol_copy = mol.__copy__()
        mol_copy.GetBondWithIdx(bond1_id).SetStereo(Chem.rdchem.BondStereo.STEREOE)
        mol_copy.GetBondWithIdx(bond2_id).SetStereo(Chem.rdchem.BondStereo.STEREOE)
        mol_copy = EZ_helper(mol_copy, [bond1_id, bond2_id])
        list_of_smiles.append(("EE", Chem.CanonSmiles(Chem.MolToSmiles(mol_copy))))

        # (E, Z)
        mol_copy = mol.__copy__()
        mol_copy.GetBondWithIdx(bond1_id).SetStereo(Chem.rdchem.BondStereo.STEREOE)
        mol_copy.GetBondWithIdx(bond2_id).SetStereo(Chem.rdchem.BondStereo.STEREOZ)
        mol_copy = EZ_helper(mol_copy, [bond1_id, bond2_id])
        list_of_smiles.append(("EZ", Chem.CanonSmiles(Chem.MolToSmiles(mol_copy))))

        # (Z, E)
        mol_copy = mol.__copy__()
        mol_copy.GetBondWithIdx(bond1_id).SetStereo(Chem.rdchem.BondStereo.STEREOZ)
        mol_copy.GetBondWithIdx(bond2_id).SetStereo(Chem.rdchem.BondStereo.STEREOE)
        mol_copy = EZ_helper(mol_copy, [bond1_id, bond2_id])
        list_of_smiles.append(("ZE", Chem.CanonSmiles(Chem.MolToSmiles(mol_copy))))

        # (Z, Z)
        mol_copy = mol.__copy__()
        mol_copy.GetBondWithIdx(bond1_id).SetStereo(Chem.rdchem.BondStereo.STEREOZ)
        mol_copy.GetBondWithIdx(bond2_id).SetStereo(Chem.rdchem.BondStereo.STEREOZ)
        mol_copy = EZ_helper(mol_copy, [bond1_id, bond2_id])
        list_of_smiles.append(("ZZ", Chem.CanonSmiles(Chem.MolToSmiles(mol_copy))))

        return list_of_smiles
    
    except:
        return []
