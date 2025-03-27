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
