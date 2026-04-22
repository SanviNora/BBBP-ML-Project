from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from collections import defaultdict


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def get_scaffold_split(X, y, smiles_list, test_frac=0.1, val_frac=0.1, seed=42):
    """
    Groups molecules by scaffold. Assigns entire scaffold groups to splits.
    Guarantees zero scaffold overlap between train and test.
    """
    scaffold_to_idxs = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        sc = get_scaffold(smi)
        scaffold_to_idxs[sc].append(i)

    scaffold_sets = sorted(scaffold_to_idxs.values(), key=len, reverse=True)

    n = len(y)
    test_cutoff = int(np.floor(test_frac * n))
    val_cutoff  = int(np.floor(val_frac  * n))

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(test_idx) < test_cutoff:
            test_idx.extend(group)
        elif len(val_idx) < val_cutoff:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    def pack(idxs):
        idxs = np.array(idxs)
        return {'X': X[idxs], 'y': y[idxs],
                'smiles': [smiles_list[i] for i in idxs]}
    return pack(train_idx), pack(val_idx), pack(idx_test)