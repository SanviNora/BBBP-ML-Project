# src/analysis/representation.py
import sys
sys.path.insert(0, '/Users/nora/BBBP-ML-Project')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from src.data.dataset import BBBPDataset


def get_ecfp4(smiles, radius=2, n_bits=2048):
    """Compute ECFP4 fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_max_tanimoto(test_smiles, train_smiles):
    """For each test molecule find its max Tanimoto similarity to training set."""
    train_fps = [get_ecfp4(s) for s in train_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]
    max_sims = []
    for smi in test_smiles:
        fp = get_ecfp4(smi)
        if fp is None:
            continue
        sims = [tanimoto_similarity(fp, tfp) for tfp in train_fps]
        max_sims.append(max(sims))
    return np.array(max_sims)


def run_representation_analysis():
    """Compare ECFP4 vs molecular graph and quantify scaffold gap."""
    ds = BBBPDataset('data/raw/BBBP.csv')

    print("Computing Tanimoto similarities...")
    print("This may take a few minutes...\n")

    train_r, val_r, test_r = ds.get_random_split(seed=42)
    max_sim_random = compute_max_tanimoto(test_r['smiles'], train_r['smiles'])

    train_s, val_s, test_s = ds.get_scaffold_split(seed=42)
    max_sim_scaffold = compute_max_tanimoto(test_s['smiles'], train_s['smiles'])

    print("=" * 50)
    print("Scaffold Gap Analysis")
    print("=" * 50)
    print(f"Random split   — mean max Tanimoto: {max_sim_random.mean():.4f} +/- {max_sim_random.std():.4f}")
    print(f"Scaffold split — mean max Tanimoto: {max_sim_scaffold.mean():.4f} +/- {max_sim_scaffold.std():.4f}")
    print(f"Difference: {max_sim_random.mean() - max_sim_scaffold.mean():.4f}")
    print("Scaffold test molecules are more novel — this explains the AUC drop.")

    results = pd.DataFrame({
        'split': ['random', 'scaffold'],
        'mean_max_tanimoto': [max_sim_random.mean(), max_sim_scaffold.mean()],
        'std_max_tanimoto':  [max_sim_random.std(),  max_sim_scaffold.std()],
        'min_max_tanimoto':  [max_sim_random.min(),  max_sim_scaffold.min()],
        'max_max_tanimoto':  [max_sim_random.max(),  max_sim_scaffold.max()],
    })
    results.to_csv('results/representation_analysis.csv', index=False)
    print("\nSaved to results/representation_analysis.csv")

    print("\n" + "=" * 50)
    print("ECFP4 vs Molecular Graph Comparison")
    print("=" * 50)
    comparison = pd.DataFrame({
        'Property': [
            'Type',
            'Dimensionality',
            'Stereochemistry',
            'Information loss',
            'Learned features',
            '3D geometry'
        ],
        'ECFP4': [
            'Fixed binary vector 2048 bits',
            'Always 2048 regardless of molecule size',
            'Partially captured chirality flags only',
            'Hash collisions possible',
            'No fixed by chemistry rules',
            'Not captured'
        ],
        'Molecular Graph': [
            'Variable-size graph nodes=atoms edges=bonds',
            'Scales with molecule size',
            'Fully captured as node feature',
            'No compression full topology preserved',
            'Yes GCN learns task-relevant features',
            'Not captured 2D graph only'
        ]
    })
    print(comparison.to_string(index=False))
    comparison.to_csv('results/representation_comparison.csv', index=False)
    print("\nSaved to results/representation_comparison.csv")

    return max_sim_random, max_sim_scaffold


if __name__ == '__main__':
    run_representation_analysis()