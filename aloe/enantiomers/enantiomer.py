from collections import OrderedDict, defaultdict

from rdkit import Chem


def enantiomer(l1, l2):
    """Check if two lists of stereo centers are enantiomers"""
    indicator = True
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        assert idx1 == idx2
        if stereo1 == stereo2:
            indicator = False
            return indicator
    return indicator


def enantiomer_helper(smiles):
    """get non-enantiomer SMILES from given list of smiles"""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    stereo_centers = [
        Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)
        for mol in filter(None, mols)
    ]
    non_enantiomers = []
    non_centers = []
    for i in range(len(stereo_centers)):
        smi = smiles[i]
        stereo = stereo_centers[i]
        indicator = True
        if len(stereo) > 0:
            for stereo_j in non_centers:
                if enantiomer(stereo_j, stereo):
                    indicator = False
                    break
        if indicator:
            non_centers.append(stereo)
            non_enantiomers.append(smi)
    return non_enantiomers


def remove_enantiomers(inpath, out):
    """Removing enantiomers for the input file
    Arguments:
        inpath: input smi
        output: output smi
    """
    with open(inpath, "r") as f:
        data = f.readlines()

    smiles = defaultdict(lambda: [])
    for line in data:
        vals = line.split()
        smi, name = vals[0].strip(), vals[1].strip().split("_")[0].strip()
        smiles[name].append(smi)

    for key, values in smiles.items():
        try:
            new_values = enantiomer_helper(values)
        except:
            new_values = values
            print(f"Enantiomers not removed for {key}", flush=True)

        smiles[key] = new_values

    with open(out, "w+") as f:
        for key, val in smiles.items():
            for i in range(len(val)):
                new_key = key + "_" + str(i)
                line = val[i].strip() + " " + new_key + "\n"
                f.write(line)
    return smiles


def no_enantiomer_helper(info1, info2):
    """Return true if info1 and info2 are enantiomers"""
    assert len(info1) == len(info2)
    for i in range(len(info1)):
        if info1[i].strip() == info2[i].strip():
            return False
    return True


def get_stereo_info(smi):
    "Return a dictionary of @@ or @  in smi"
    dct = {}
    regex1 = re.compile("[^@]@[^@]")
    regex2 = re.compile("@@")

    # match @
    for m in regex1.finditer(smi):
        dct[m.start() + 1] = "@"

    # match  @@
    for m in regex2.finditer(smi):
        dct[m.start()] = "@@"

    dct2 = OrderedDict(sorted(dct.items()))
    return dct2


def no_enantiomer(smi, smiles):
    """Return True if there is no enantiomer for smi in smiles"""

    stereo_infoi = list(get_stereo_info(smi).values())
    for i in range(len(smiles)):
        tar = smiles[i]
        if tar != smi:
            stereo_infoj = list(get_stereo_info(tar).values())
            if no_enantiomer_helper(stereo_infoi, stereo_infoj):
                return False
    return True


def create_enantiomer(smi):
    """Create an enantiomer SMILES for input smi"""
    stereo_info = get_stereo_info(smi)
    new_smi = ""
    # for key in stereo_info.keys():
    #     val = stereo_info[key]
    #     if val == '@':
    keys = list(stereo_info.keys())
    if len(keys) == 1:
        key = keys[0]
        val = stereo_info[key]
        if val == "@":
            new_smi += smi[:key]
            new_smi += "@@"
            new_smi += smi[(key + 1) :]
        elif val == "@@":
            new_smi += smi[:key]
            new_smi += "@"
            new_smi += smi[(key + 2) :]
        else:
            raise ValueError("Invalid %s" % smi)
        return new_smi

    for i in range(len(keys)):
        if i == 0:
            key = keys[i]
            new_smi += smi[:key]
        else:
            key1 = keys[i - 1]
            key2 = keys[i]
            val1 = stereo_info[key1]
            if val1 == "@":
                new_smi += "@@"
                new_smi += smi[int(key1 + 1) : key2]
            elif val1 == "@@":
                new_smi += "@"
                new_smi += smi[int(key1 + 2) : key2]
    val2 = stereo_info[key2]
    if val2 == "@":
        new_smi += "@@"
        new_smi += smi[int(key2 + 1) :]
    elif val2 == "@@":
        new_smi += "@"
        new_smi += smi[int(key2 + 2) :]
    return new_smi
