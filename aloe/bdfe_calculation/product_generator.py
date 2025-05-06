import pandas as pd
from typing import Callable, Optional
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction


from aloe.file_utils import read_csv_dict
from aloe.isomer_generation.isomer_engine import enumerate_EZ_diimine

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
        
def determine_reaction_from_key(key: str) -> ChemicalReaction:
    r"""
    Determines the reaction to use to generate products from a reactant.

    Args:
        key (str): The key of the reactant (type of substructure, e.g. "o-diol").

    Returns:
        reaction (ChemicalReaction): The reaction to use to generate products from a reactant.
    """
    other_key, _ = translate_key(key)
    reaction_smarts = SUBSTRUCTURES[key] + ">>" + SUBSTRUCTURES[other_key]
    reaction = AllChem.ReactionFromSmarts(reaction_smarts)
    return reaction


def get_products_from_reactant(name: str, smi: str, key: str, reaction: ChemicalReaction):
    r"""
    Creates products from a reactant using a reaction.
    
    Args:
        name (str): The name of the reactant.
        smi (str): The SMILES string of the reactant.
        key (str): The key of the reactant (type of substructure, e.g. "o-diol").
        reaction (ChemicalReaction): The reaction to use to create products.
    
    Returns:
        products (list): A list of products.
        names (list): A list of names for the products.
    """
    mol = Chem.MolFromSmiles(smi)
    product_key, _ = translate_key(key)
    products = []
    for i in reaction.RunReactants((mol,)):
        product = i[0]
        try:
            Chem.SanitizeMol(product)
            products.append(Chem.CanonSmiles(Chem.MolToSmiles(product)))
        except:
            continue
    products = list(set(products))
    names = ['_'.join([name, key, 'product'+str(i), product_key]) for i in range(len(products))]
    return products, names

def generate_products(input_file: str, output_file: str, key: str, reaction: ChemicalReaction, post_process_function: Optional[Callable] = None):
    r"""
    Generates products from the input file using reactions.
    Arguments:
        input_file (str): Path to the input file containing SMILES strings.
        output_file (str): Path to the output file containing the generated products.
        key (str): The key of the reactant (type of substructure, e.g. "o-diol").
        reaction (ChemicalReaction): The reaction to use to create products.
        post_process_function (Callable): A function to post-process the products.
    Returns:
        failed (list): A list of tuples containing the keys and SMILES strings for which no products were generated.
    """
    reaction.Initialize()

    products = []
    names = []
    failed = []

    data = read_csv_dict(input_file)
    for name, smi in data.items():
        these_products, these_names = get_products_from_reactant(name=name, smi=smi, key=key, reaction=reaction)
        if len(these_products) == 0:
            failed.append((name, smi))
        else:
            products.extend(these_products)
            names.extend(these_names)
    
    df = pd.DataFrame({'Name': names, 'SMILES': products})

    if post_process_function is not None:
        df = post_process_function(df)

    df.to_csv(output_file, index=False)

    return failed


def clean_up_diamine(product: Mol) -> str:
    r"""
    Cleans up a diamine by removing the hydrogens on the nitrogen atoms.
    """
    product_copy = Chem.RWMol(product)
    to_be_removed = []
    for atom in product_copy.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 and atom.GetNumExplicitHs() == 2:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    to_be_removed.append(neighbor.GetIdx())
    for idx in sorted(to_be_removed, reverse=True):
        product_copy.RemoveAtom(idx)
    return Chem.CanonSmiles(Chem.MolToSmiles(product_copy.GetMol()))

def diamine_hook(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Cleans up a diamine by removing the hydrogens on the nitrogen atoms.
    """
    mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    cleaned_smies = [clean_up_diamine(mol) for mol in mols]
    df['SMILES'] = cleaned_smies
    return df


def diimine_goes_EZ_hook(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Enumerates the EZ isomers of a diimine.
    """
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in df['SMILES']]

    name_col = df.columns[0] if df.columns[0] != 'SMILES' else df.columns[1]
    names = df[name_col].values

    new_names, new_smiles = [], []
    for name, mol in zip(names, mols):
        list_of_smiles = enumerate_EZ_diimine(mol)
        new_names.extend([f"{name}-{code}" for code, _ in list_of_smiles])
        new_smiles.extend([smi for _, smi in list_of_smiles])
    df = pd.DataFrame({'Name': new_names, 'SMILES': new_smiles})
    return df
