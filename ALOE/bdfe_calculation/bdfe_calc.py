import csv

from rdkit import Chem

from aloe.backend import generate_stereoisomers
from aloe.bdfe_calculation.product_generator import generate_products
from aloe.file_utils import make_output_name
from aloe.isomer_generation.isomer_engine import rd_enumerate_EZ_isomer


def products_pipeline(input_file):
    r"""
    pipeline to generate products from the input file using the product generator.

    Arguments:
        input_file (str): Path to the input file containing SMILES strings. (A chunk file)
    """
    reactants = generate_stereoisomers(input_file)  # stereoisomerization

    products = make_output_name(input_file, "products", ".csv")

    products, failed = generate_products(
        reactants, products
    )  # generate products from reactants

    # enumerate stereoisomers for the products
    output_file = ".".split(products)[0] + "_isomers.csv"
    engine = rd_enumerate_EZ_isomer(
        csv=products,
        enumerated_csv=output_file,
    )

    output_file = engine.run()

    return output_file, failed


def bdfe_calculator(reactant_file, product_file):
    r"""
    pipeline to calculate BDFE from reactants and products.

    Arguments:
        reactant_file (str): Path to the reactant file (with stereoisomers).
        product_file (str): Path to the product file (with stereoisomers).

    Returns:
        str: path to the final BDFE calculation file.
    """
    output_file = make_output_name(reactant_file, "bdfe", ".csv")

    reactant_supplier = Chem.SDMolSupplier(reactant_file)
    product_supplier = Chem.SDMolSupplier(product_file)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Reactant", "Product", "Reaction Details", "BDFE (kcal/mol"])

        for rmol in reactant_supplier:
            # Getting reactant information
            reactant_name = rmol.GetProp("_Name") if rmol.HasProp("_Name") else None
            reactant_smi = Chem.MolToSmiles(rmol, isomericSmiles=True)
            reatant_bdfe = (
                rmol.GetProp("G_hartree") if rmol.HasProp("G_hartree") else None
            )

            for pmol in product_supplier:
                # Getting product information
                product_info = pmol.GetProp("_Name") if pmol.HasProp("_Name") else None
                product_info = "-".split(product_info)

                if reactant_name == product_info[0]:
                    product_smi = Chem.MolToSmiles(pmol, isomericSmiles=True)
                    product_bdfe = (
                        pmol.GetProp("G_hartree") if pmol.HasProp("G_hartree") else None
                    )

                    bdfe = None

                    if reatant_bdfe is not None and product_bdfe is not None:
                        bdfe = product_bdfe - reatant_bdfe

                        # Convert to kcal/mol from Hartree
                        bdfe = float(bdfe) * 627.509

                    writer.writerow(
                        [
                            reactant_smi,
                            product_smi,
                            "-".join(product_info),
                            str(bdfe) if bdfe is not None else "Calculation Failed",
                        ]
                    )

                else:
                    continue

    return output_file


def write_failed_reactants(input_file, failed_reactants):
    r"""
    Write the failed reactants to a new file for further analysis.
    Arguments:
        input_file (str): Path to the input file containing reactants.
        failed_reactants (list): List of SMILES strings for failed reactants.
    Returns:
        str: Path to the file containing failed reactants.
    """

    if failed_reactants is None or len(failed_reactants) == 0:
        # No failed reactants to write
        return None

    output_file = make_output_name(input_file, "failed_reactants", ".csv")

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "SMILES"])
        for (key, smi) in failed_reactants:
            writer.writerow(key, smi)

    return output_file
