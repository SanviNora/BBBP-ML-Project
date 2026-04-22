import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from config import SMILES_COL, LABEL_COL, ECFP_RADIUS, ECFP_NBITS, SEED


def smiles_to_ecfp4(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Converts a single SMILES string to an ECFP4 binary float vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


class BBBPDataset:
    """
    Loads BBBP.csv, computes ECFP4 fingerprints, returns train/val/test splits.
    This is the SINGLE data source that all models import from.
    """

    def __init__(self, csv_path: str, radius: int = ECFP_RADIUS,
                 n_bits: int = ECFP_NBITS, seed: int = SEED):
        self.seed = seed
        df = pd.read_csv(csv_path)
        fps, labels, smiles = [], [], []
        dropped = 0
        for _, row in df.iterrows():
            fp = smiles_to_ecfp4(row[SMILES_COL], radius, n_bits)
            if fp is None:
                dropped += 1
                continue
            fps.append(fp)
            labels.append(int(row[LABEL_COL]))
            smiles.append(row[SMILES_COL])
        print(f'Loaded {len(fps)} molecules ({dropped} dropped as invalid SMILES)')
        self.X      = np.stack(fps)
        self.y      = np.array(labels)
        self.smiles = smiles

    def get_random_split(self, seed: int) -> tuple:
        """Returns (train, val, test) as dicts with keys X, y, smiles."""
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(self.y))
        idx_tv, idx_test  = train_test_split(idx, test_size=0.10, random_state=seed)
        idx_train, idx_val = train_test_split(idx_tv, test_size=0.111, random_state=seed)
        def pack(i):
            return {'X': self.X[i], 'y': self.y[i],
                    'smiles': [self.smiles[j] for j in i]}
        return pack(idx_train), pack(idx_val), pack(idx_test)

    def get_scaffold_split(self, seed: int) -> tuple:
        """Bemis-Murcko scaffold split — zero overlap between train and test scaffolds."""
        from src.data.splits import get_scaffold_split
        return get_scaffold_split(self.X, self.y, self.smiles, seed=seed)
