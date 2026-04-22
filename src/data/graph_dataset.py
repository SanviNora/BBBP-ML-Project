import torch
from torch_geometric.data import Data
from rdkit import Chem

def get_atom_features(atom):
    """
    Extract a 9-dimensional atom feature vector.

    Features:
        1. Atomic number
        2. Degree (number of bonds)
        3. Formal charge
        4. Chiral tag
        5. Number of hydrogen atoms
        6. Hybridization type
        7. Is aromatic (0 or 1)
        8. Is in a ring (0 or 1)
        9. Total valence
    """
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalValence(),
    ]


def smiles_to_graph(smiles: str) -> Data:
    """
    Convert a SMILES string into a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string of the molecule.
                e.g. caffeine: "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

    Returns:
        PyG Data object with:
            x          -- node feature matrix, shape [num_atoms, 9]
            edge_index -- COO edge index,       shape [2, num_bonds * 2]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    # Build node feature matrix: shape [num_atoms, 9]
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Build edge index: each bond is added in both directions (undirected graph)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
