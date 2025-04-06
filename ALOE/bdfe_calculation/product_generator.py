import csv

from rdkit import Chem
from rdkit.Chem import AllChem

from aloe.file_utils import read_csv_dict

# Carbons in central aryl ring numbered 1-6, hetero atoms numbered 7 & 8.
SUBSTRUCTURES = {
    # Hydrogenated substructures
    "o-diamines": "[NH2:7]-[c:1]1[c:2](-[NH2:8])[c:3][c:4][c:5][c:6]1",
    "p-diamines": "[NH2:7]-[c:1]1[c:2][c:3][c:4](-[NH2:8])[c:5][c:6]1",
    "o-diols": "[OH:7]-[c:1]1[c:2](-[OH:8])[c:3][c:4][c:5][c:6]1",
    "p-diols": "[OH:7]-[c:1]1[c:2][c:3][c:4](-[OH:8])[c:5][c:6]1",
    "o-dithiols": "[SH:7]-[c:1]1[c:2](-[SH:8])[c:3][c:4][c:5][c:6]1",
    "p-dithiols": "[SH:7]-[c:1]1[c:2][c:3][c:4](-[SH:8])[c:5][c:6]1",
    "pyridine-2,3-diamines": "[NH2:7]-[c:1]1[c:2](-[NH2:8])[n:3][c:4][c:5][c:6]1",
    "pyridine-3,4-diamines": "[NH2:7]-[c:1]1[c:2](-[NH2:8])[c:3][n:4][c:5][c:6]1",
    "pyridine-2,3-diols": "[OH:7]-[c:1]1[c:2](-[OH:8])[n:3][c:4][c:5][c:6]1",
    "pyridine-3,4-diols": "[OH:7]-[c:1]1[c:2](-[OH:8])[c:3][n:4][c:5][c:6]1",
    "pyridine-2,3-dithiols": "[SH:7]-[c:1]1[c:2](-[SH:8])[n:3][c:4][c:5][c:6]1",
    "pyridine-3,4-dithiols": "[SH:7]-[c:1]1[c:2](-[SH:8])[c:3][n:4][c:5][c:6]1",
    "furan-2,3-diamines": "[o:1]1[c:2](-[NH2:6])[c:3](-[NH2:7])[c:4][c:5]1",
    "furan-2,5-diamines": "[o:1]1[c:2](-[NH2:6])[c:3][c:4][c:5](-[NH2:7])1",
    "thiophene-2,3-diamines": "[s:1]1[c:2](-[NH2:6])[c:3](-[NH2:7])[c:4][c:5]1",
    "thiophene-2,5-diamines": "[s:1]1[c:2](-[NH2:6])[c:3][c:4][c:5](-[NH2:7])1",
    "furan-2,3-diols": "[o:1]1[c:2](-[OH:6])[c:3](-[OH:7])[c:4][c:5]1",
    "furan-2,5-diols": "[o:1]1[c:2](-[OH:6])[c:3][c:4][c:5](-[OH:7])1",
    "thiophene-2,3-diols": "[s:1]1[c:2](-[OH:6])[c:3](-[OH:7])[c:4][c:5]1",
    "thiophene-2,5-diols": "[s:1]1[c:2](-[OH:6])[c:3][c:4][c:5](-[OH:7])1",
    "furan-2,3-dithiols": "[o:1]1[c:2](-[SH:6])[c:3](-[SH:7])[c:4][c:5]1",
    "furan-2,5-dithiols": "[o:1]1[c:2](-[SH:6])[c:3][c:4][c:5](-[SH:7])1",
    "thiophene-2,3-dithiols": "[s:1]1[c:2](-[SH:6])[c:3](-[SH:7])[c:4][c:5]1",
    "thiophene-2,5-dithiols": "[s:1]1[c:2](-[SH:6])[c:3][c:4][c:5](-[SH:7])1",
    # Dehydrogenated substructures
    "o-diimines": "[NH:7]=[C:1]-1(-[C:2](=[NH:8])(-[C:3]=[C:4]-[C:5]=[C:6]-1))",
    "p-diimines": "[NH:7]=[C:1]-1(-[C:2](=[C:3]-[C:4](=[NH:8])-[C:5]=[C:6]-1))",
    "o-quinones": "[O:7]=[C:1]-1(-[C:2](=[O:8])(-[C:3]=[C:4]-[C:5]=[C:6]-1))",
    "p-quinones": "[O:7]=[C:1]-1(-[C:2](=[C:3]-[C:4](=[O:8])-[C:5]=[C:6]-1))",
    "o-dithiones": "[S:7]=[C:1]-1(-[C:2](=[S:8])(-[C:3]=[C:4]-[C:5]=[C:6]-1))",
    "p-dithiones": "[S:7]=[C:1]-1(-[C:2](=[C:3]-[C:4](=[S:8])-[C:5]=[C:6]-1))",
    "pyridine-2,3-diimines": "[NH:7]=[C:1]-1(-[C:2](=[NH:8])(-[N:3]=[C:4]-[C:5]=[C:6]-1))",
    "pyridine-3,4-diimines": "[NH:7]=[C:1]-1(-[C:2](=[NH:8])(-[C:3]=[N:4]-[C:5]=[C:6]-1))",
    "pyridine-2,3-diones": "[O:7]=[C:1]-1(-[C:2](=[O:8])(-[N:3]=[C:4]-[C:5]=[C:6]-1))",
    "pyridine-3,4-diones": "[O:7]=[C:1]-1(-[C:2](=[O:8])(-[C:3]=[N:4]-[C:5]=[C:6]-1))",
    "pyridine-2,3-dithiones": "[S:7]=[C:1]-1(-[C:2](=[S:8])(-[N:3]=[C:4]-[C:5]=[C:6]-1))",
    "pyridine-3,4-dithiones": "[S:7]=[C:1]-1(-[C:2](=[S:8])(-[C:3]=[N:4]-[C:5]=[C:6]-1))",
    "furan-2,3-diimines": "[O:1]-1(-[C:2](=[NH:6])(-[C:3](=[NH:7])-[C:4]=[C:5]-1))",
    "furan-2,5-diimines": "[O:1]-1(-[C:2](=[NH:6])(-[C:3]=[C:4]-[C:5](=[NH:7])-1))",
    "thiophene-2,3-diimines": "[S:1]-1(-[C:2](=[NH:6])(-[C:3](=[NH:7])-[C:4]=[C:5]-1))",
    "thiophene-2,5-diimines": "[S:1]-1(-[C:2](=[NH:6])(-[C:3]=[C:4]-[C:5](=[NH:7])-1))",
    "furan-2,3-diones": "[O:1]-1(-[C:2](=[O:6])(-[C:3](=[O:7])-[C:4]=[C:5]-1))",
    "furan-2,5-diones": "[O:1]-1(-[C:2](=[O:6])(-[C:3]=[C:4]-[C:5](=[O:7])-1))",
    "thiophene-2,3-diones": "[S:1]-1(-[C:2](=[O:6])(-[C:3](=[O:7])-[C:4]=[C:5]-1))",
    "thiophene-2,5-diones": "[S:1]-1(-[C:2](=[O:6])(-[C:3]=[C:4]-[C:5](=[O:7])-1))",
    "furan-2,3-dithiones": "[O:1]-1(-[C:2](=[S:6])(-[C:3](=[S:7])-[C:4]=[C:5]-1))",
    "furan-2,5-dithiones": "[O:1]-1(-[C:2](=[S:6])(-[C:3]=[C:4]-[C:5](=[S:7])-1))",
    "thiophene-2,3-dithiones": "[S:1]-1(-[C:2](=[S:6])(-[C:3](=[S:7])-[C:4]=[C:5]-1))",
    "thiophene-2,5-dithiones": "[S:1]-1(-[C:2](=[S:6])(-[C:3]=[C:4]-[C:5](=[S:7])-1))",
}


pairs = [
    ("o-diol", "o-quinone"),
    ("p-diol", "p-quinone"),
    ("2,3-diol", "2,3-dione"),
    ("2,5-diol", "2,5-dione"),
    ("3,4-diol", "3,4-dione"),
    ("diamine", "diimine"),
    ("dithiol", "dithione"),
]


def translate_key(key):
    for red, ox in pairs:
        if red in key:
            return key.replace(red, ox), "reduced"
        if ox in key:
            return key.replace(ox, red), "oxidized"


def get_products_from_reactant(smi, reactions):
    mol = Chem.MolFromSmiles(smi)
    products = []
    for key in reactions:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(SUBSTRUCTURES[key])):
            rxn = reactions[key]
            products_smiles = []
            for i in rxn.RunReactants((mol,)):
                product = i[0]
                try:
                    Chem.SanitizeMol(product)
                    products_smiles += Chem.MolToSmiles(product)
                except:
                    continue
            products += products_smiles
    return products

    # you need to figure out the names for the products here


def generate_products(input_file, output_file):
    r"""
    Generates products from the input file using the defined substructures and their oxidization/reduction reactions.
    Arguments:
        input_file (str): Path to the input file containing SMILES strings.
    Returns:
        output_file (str): Path to the output file containing the generated products.
        failed (list): A list of tuples containing the keys and SMILES strings for which no products were generated.
    """
    oxidization_reactions = {}
    reduction_reactions = {}

    for key in SUBSTRUCTURES:
        new_key, oxidation_state = translate_key(key)
        match oxidation_state:
            case "oxidized":
                reduction_reactions[key] = AllChem.ReactionFromSmarts(
                    f"{SUBSTRUCTURES[key]}>>{SUBSTRUCTURES[new_key]}"
                )
            case "reduced":
                oxidization_reactions[key] = AllChem.ReactionFromSmarts(
                    f"{SUBSTRUCTURES[key]}>>{SUBSTRUCTURES[new_key]}"
                )

    for key in oxidization_reactions:
        oxidization_reactions[key].Initialize()
    for key in reduction_reactions:
        reduction_reactions[key].Initialize()

    data = read_csv_dict(input_file)

    no_products = []
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "SMILES"])
        for key, val in data.items():
            # TODO: figure out which reactions to run
            products = get_products_from_reactant(val, oxidization_reactions)
            products.extend(get_products_from_reactant(val, reduction_reactions))

            if len(products) == 0:
                no_products.append((key, val))
            else:
                for i, reactions in enumerate(products):
                    for j, smi in enumerate(reactions):
                        # TODO: better naming scheme for "reaction", and of specific product name
                        writer.writerow([f"{key}-reaction{i}-diimine{j}", smi])

    return output_file, no_products
