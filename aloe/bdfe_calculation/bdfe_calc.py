import pandas as pd
from rdkit import Chem

from aloe.bdfe_calculation.product_generator import (
    determine_reaction_from_key,
    diamine_hook,
    diimine_goes_EZ_hook,
    generate_products,
)
from aloe.file_utils import make_output_name
from aloe.frontend import ConformerConfig, OptConfig, RankConfig, ThermoConfig

HARTREE_TO_KCAL = 627.5096080305927
G_H = -0.51016097


def get_G(file: str) -> pd.DataFrame:
    r"""
    Get the free energies from an .sdf file.

    Args:
        file (str): Path to the .sdf file.

    Returns:
        pd.DataFrame: A dataframe containing the free energies.
    """

    df = pd.DataFrame(columns=["Name", "SMILES", "G_hartree"])

    with Chem.SDMolSupplier(file) as supplier:
        for mol in supplier:
            name, smi = mol.GetProp("_Name"), Chem.MolToSmiles(mol)
            G = float(mol.GetProp("G_hartree"))
            df.loc[len(df)] = [name, smi, G]

    return df


def get_BDFE(
    reduced_form_file: str,
    oxidized_form_file: str,
    num_Hs: int = 2,
    use_reduced_names: bool = True,
) -> pd.DataFrame:
    r"""
    Args:
        reduced_form_file (str): Path to the reduced form .sdf file.
        oxidized_form_file (str): Path to the oxidized form .sdf file.
        num_Hs (int): Number of hydrogens to remove from the reduced form.
        use_reduced_names (bool): Whether to use the names of the reduced forms as the base names.

    Returns:
        pd.DataFrame: A dataframe containing the average BDFE (kcal/mol).
    """

    reduced_form_df = get_G(reduced_form_file)
    reduced_form_df.columns = ["Name_reduced", "SMILES_reduced", "G_hartree_reduced"]
    oxidized_form_df = get_G(oxidized_form_file)
    oxidized_form_df.columns = [
        "Name_oxidized",
        "SMILES_oxidized",
        "G_hartree_oxidized",
    ]

    if use_reduced_names:
        reduced_form_df["base_name"] = reduced_form_df["Name_reduced"].copy()
        oxidized_form_df["base_name"] = oxidized_form_df["Name_oxidized"].apply(
            lambda x: x.split("_")[0]
        )

    else:
        reduced_form_df["base_name"] = reduced_form_df["Name_reduced"].apply(
            lambda x: x.split("_")[0]
        )
        oxidized_form_df["base_name"] = oxidized_form_df["Name_oxidized"].copy()

    big_df = pd.merge(reduced_form_df, oxidized_form_df, on="base_name", how="inner")
    big_df["BDFE"] = (
        (big_df["G_hartree_oxidized"] - big_df["G_hartree_reduced"] + num_Hs * G_H)
        * HARTREE_TO_KCAL
        / num_Hs
    )

    return big_df


def write_failed_reactants(filename: str, failed_reactants: list) -> None:
    r"""
    Write the failed reactants to a new file for further analysis.

    Args:
        filename (str): Path to the file to write the failed reactants to.
        failed_reactants (list): List of tuples containing the reactant name
            and SMILES string for failed reactants.

    Returns:
        (str): Path to the file containing failed reactants.
    """

    df = pd.DataFrame(failed_reactants, columns=["Name", "SMILES"])
    df.to_csv(filename, index=False)


def calculate_bdfes_from_reduced_forms(
    input_file: str, key: str
) -> tuple[str, str | None]:
    r"""
    Generates the products from a list of reactants and calculates the change
    in bond dissociation free energy (BDFE) for each reaction.

    Args:
        input_file (str): Path to the input .csv file containing reactants.
        key (str): The key of the reactant (type of substructure, e.g. "o-diol").

    Returns:
        paths (tuple[str, str or None]): Path to the output .csv file containing the BDFE calculations
            and the path to the .csv file containing the failed reactants (if any).
    """

    reaction = determine_reaction_from_key(key)
    products_csv = make_output_name(input_file, "products", ".csv")

    if "diamine" in key:
        post_process_function = diimine_goes_EZ_hook
    elif "diimine" in key:
        post_process_function = diamine_hook

    failed_reactants = generate_products(
        input_file=input_file,
        output_file=products_csv,
        key=key,
        reaction=reaction,
        post_process_function=post_process_function,
    )

    def engine_helper(input_file) -> str:
        r"""
        Helper function to run the aloe pipeline on the input file.
        """
        from aloe import aloe

        engine = aloe(input_file, use_gpu=True)
        engine.add_step(ConformerConfig())
        engine.add_step(OptConfig())
        engine.add_step(RankConfig(k=1))
        engine.add_step(ThermoConfig())
        output_file = engine.run()
        return output_file

    reduced_form_file = engine_helper(input_file)
    oxidized_form_file = engine_helper(products_csv)

    input_file_basename = input_file.split(".")[0]

    bdfe_output_file = input_file_basename + "_bdfes_out.csv"
    get_BDFE(reduced_form_file, oxidized_form_file).to_csv(bdfe_output_file)

    if len(failed_reactants) > 0:
        failed_output_file = input_file_basename + "_failed_reactants.csv"
        failed_file = write_failed_reactants(failed_output_file, failed_reactants)
    else:
        failed_file = None

    return bdfe_output_file, failed_file
