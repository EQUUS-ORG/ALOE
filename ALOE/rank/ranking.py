from typing import List

import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm

from aloe.model_validation import ev2kcalpermol, hartree2ev
from aloe.rank.rank_utils import check_connectivity, filter_unique


class ranking(object):
    """
    Finding 3D structures that satisfy the user-defined requirements.

    Arguments:
        input_path: An SDF file that contains all isomers.
        out_path: An SDF file that stores the optimical structures.
        k: Outputs the top-k structures for each SMILES.
        window: Outputs the structures whose energies are within
                window (kcal/mol) from the lowest energy
    Returns:
        None
    """

    def __init__(
        self, input_path, out_path, threshold, k=None, window=None, encoded=False
    ):
        r"""
        Args:
            encoded: Whether the molecule names are encoded. An encoded name would look like: 0_1_2. We want 0 instead.
        """
        self.input_path = input_path
        self.out_path = out_path
        self.threshold = threshold
        self.atomic_number2symbol = {
            1: "H",
            5: "B",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            14: "Si",
            15: "P",
            16: "S",
            17: "Cl",
            32: "Ge",
            33: "As",
            34: "Se",
            35: "Br",
            51: "Sb",
            52: "Te",
            53: "I",
        }
        self.k = k
        self.window = window
        self.encoded = encoded

        if window is not None:
            assert window >= 0
            self.window_eV = window / ev2kcalpermol  # convert energy window into eV

    def get_name(self, name):
        if self.encoded:
            return name.strip().split("_")[0].strip()
        return name

    @staticmethod
    def add_relative_e(list0):
        """Adding relative energies compared with lowest-energy structure

        Argument:
            list: a list of tuple (idx, name, energy)

        Return:
            list of tuple (idx, name, energy, relative_energy)
        """
        list0_ = []
        _, _, e_m = list0[0]
        for idx_name_e in list0:
            idx_i, name_i, e_i = idx_name_e
            e_relative = e_i - e_m
            list0_.append((idx_i, name_i, e_i, e_relative))
        return list0_

    def top_k(self, df_group: pd.DataFrame, k: int = 1) -> List[Chem.Mol]:
        """
        Select the k structures with the lowest energies for the given molecule.

        Args:
            df_group: a small dataframe that contains optimized conformers for the same molecule with columns ["name", "energy", "mol"]
            k: the number of structures to be selected
        Returns:
            out_mols: a list of RDKit molecules with the lowest energies
        """
        names = list(df_group["name"])
        assert len(set(names)) == 1

        group = df_group.sort_values(by=["energy"], ascending=True).reset_index(
            drop=True
        )
        out_mols = filter_unique(list(group["mol"]), threshold=self.threshold, k=k)

        if len(out_mols) == 0:
            name = names[0].split("_")[0].strip()
            print(f"No structure converged for {name}.", flush=True)
        else:
            # Adding relative energies
            ref_energy = float(out_mols[0].GetProp("E_tot"))
            for mol in out_mols:
                my_energy = float(mol.GetProp("E_tot"))
                rel_energy = my_energy - ref_energy
                mol.SetProp("E_rel(eV)", str(rel_energy))
        return out_mols

    def top_window_df(self, df_group: pd.DataFrame, window: float) -> pd.DataFrame:
        r"""
        A simple helper function to return the lowest energy structures within a window.
        """

        assert len(df_group["name"].unique()) == 1

        group = df_group.sort_values(by=["energy"], ascending=True).reset_index(
            drop=True
        )
        minimum = group["energy"].min()
        maximum = window + minimum
        group = group.loc[group["energy"] <= maximum].copy()
        return group

    def top_window(self, df_group: pd.DataFrame, window: float) -> List[Chem.Mol]:
        """
        Given a group of energy_name_idxes,
        return all (idx, name, e) tuples whose energies are within
        window (eV) from the lowest energy. Unit table is based on:
        http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
        """

        assert len(df_group["name"].unique()) == 1

        group = self.top_window_df(df_group=df_group, window=window)

        out_mols_raw = filter_unique(list(group["mol"]), self.threshold)
        out_mols = []

        if len(out_mols_raw) == 0:
            name = self.get_name(group.iloc[0]["mol"].GetProp("_Name"))
            print(f"No structure converged for {name}.", flush=True)
        else:
            ref_energy = group["energy"].min()
            for mol in out_mols_raw:
                my_energy = float(mol.GetProp("E_tot"))
                rel_energy = my_energy - ref_energy
                mol.SetProp("E_rel(eV)", str(rel_energy))
                out_mols.append(mol)
        return out_mols

    def run(self) -> List[Chem.Mol]:
        """
        Lowest-energy structures will be stored in self.out_path.
        """
        print("Selecting structures that satisfy the requirements...", flush=True)
        results = []

        data2 = Chem.SDMolSupplier(self.input_path, removeHs=False)
        mols, names, energies = [], [], []
        for mol in data2:
            try:
                if (
                    (mol is not None)
                    and (mol.GetProp("Converged").lower() == "true")
                    and check_connectivity(mol)
                ):  # Verify convergence and correct connectivity
                    mols.append(mol)
                    names.append(self.get_name(mol.GetProp("_Name")))
                    energies.append(float(mol.GetProp("E_tot")))
            except:
                pass

        df = pd.DataFrame({"name": names, "energy": energies, "mol": mols})

        df2 = df.groupby("name")
        for group_name in tqdm(df2.indices):
            df_group = df2.get_group(group_name)
            print(
                f"Processing {len(df_group)} conformers for {group_name}...", flush=True
            )
            df_group = df_group.sort_values(by=["energy"], ascending=True).reset_index(
                drop=True
            )
            if self.window:
                df_group = self.top_window_df(df_group=df_group, window=self.window)
            if self.k:
                top_results = self.top_k(df_group=df_group, k=self.k)
            elif self.window:
                top_results = self.top_window(df_group=df_group, window=self.window)
            else:
                raise ValueError(
                    (
                        "Either k or window needs to be specified."
                        "Usually, setting '--k=1' satisfies most needs."
                    )
                )
            results += top_results

        with Chem.SDWriter(self.out_path) as f:
            for mol in results:
                # Change the energy unit from eV back to Hartree
                mol.SetProp("E_tot", str(float(mol.GetProp("E_tot")) / hartree2ev))
                mol.SetProp(
                    "E_rel(kcal/mol)",
                    str(float(mol.GetProp("E_rel(eV)")) * ev2kcalpermol),
                )
                mol.ClearProp("E_rel(eV)")
                # Remove _ in the molecule title if it is encoded
                mol.SetProp("_Name", self.get_name(mol.GetProp("_Name")))
                f.write(mol)
        return self.out_path
