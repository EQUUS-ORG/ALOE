
import os
from aloe.isomer_generation.isomer_engine import rd_enumerate_EZ_isomer
from aloe.backend import generate_stereoisomers
from aloe.bdfe_calculation.product_generator import generate_products
from aloe.file_utils import make_output_name


def products_pipeline(input_file):
    r"""
    pipeline to generate products from the input file using the product generator.
    
    Arguments:
        input_file (str): Path to the input file containing SMILES strings. (A chunk file)
    """
    reactants = generate_stereoisomers(input_file) # stereoisomerization
    
    products = make_output_name(input_file, "products", ".csv") 
    
    products, failed = generate_products(reactants, products) # generate products from reactants
    
    # enumerate stereoisomers for the products
    output_file = ".".split(products)[0]+ "_isomers.csv"
    engine = rd_enumerate_EZ_isomer(
        csv=products,
        enumerated_csv=output_file,
    )

    output_file = engine.run()
    
    
    return output_file, failed