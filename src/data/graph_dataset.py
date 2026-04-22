import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from src.data.dataset import BBBPDataset

HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

BOND_TYPE_LIST = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

def get_atom_features(atom):
    """
    9-dimensional atom feature vector:
        1. Atomic number
        2. Degree
        3. Formal charge
        4. Chiral tag
        5. Number of Hs
        6. Hybridization (one-hot index into HYBRIDIZATION_LIST, -1 if unknown)
        7. Is aromatic
        8. Is in ring
        9. Total valence
    """
    hybridization = atom.GetHybridization()
    hyb_idx = HYBRIDIZATION_LIST.index(hybridization) if hybridization in HYBRIDIZATION_LIST else -1

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        hyb_idx,
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalValence(),
    ]


def get_edge_features(bond):
    """
    4-dimensional edge feature vector:
        1. Bond type (one-hot index: single=0, double=1, triple=2, aromatic=3)
        2. Is conjugated
        3. Is in ring
        4. Is stereo (0 if STEREONONE, 1 otherwise)
    """
    bond_type = bond.GetBondType()
    bond_type_idx = BOND_TYPE_LIST.index(bond_type) if bond_type in BOND_TYPE_LIST else -1

    return [
        bond_type_idx,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo() != rdchem.BondStereo.STEREONONE),
    ]


def smiles_to_graph(smiles: str) -> Data:
    """
    Convert a SMILES string into a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        PyG Data object with:
            x          -- node features,  shape [num_atoms, 9]
            edge_index -- COO edge index, shape [2, num_bonds * 2]
            edge_attr  -- edge features,  shape [num_bonds * 2, 4]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    # Node features
    x = torch.tensor(
        [get_atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    # Edge index + edge features (undirected: each bond added twice)
    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_edge_features(bond)
        edges += [[i, j], [j, i]]
        edge_feats += [feat, feat]

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class BBBPGraphDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        
        # Reuse Nora's loader to get clean SMILES + labels
        base = BBBPDataset(csv_path)
        
        self.graphs = []
        skipped = 0

        for smiles, label in zip(base.smiles, base.y):
            try:
                graph = smiles_to_graph(smiles)
                graph.y = torch.tensor([label], dtype=torch.long)
                self.graphs.append(graph)
            except ValueError:
                skipped += 1

        print(f"Loaded {len(self.graphs)} graphs, skipped {skipped} invalid SMILES.")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]