##-----GCN_Model-----##
#import sys
#print(sys.executable)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
import numpy as np

class GCNModel(torch.nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    """
    def __init__(self, in_channels=9, hidden_channels=64, out_channels=2, 
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)  # shape [N]


class GCNBBB:
    """
    Wrapper around GCNModel with fit(), predict(), predict_proba().
    """
    def __init__(self, hidden_channels=64, num_layers=2, dropout=0.0,
                 lr=1e-3, epochs=200, patience=20, batch_size=32,
                 checkpoint_path="best_gcn.pt",use_class_weight=False):
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.use_class_weight = use_class_weight

    def fit(self, train_graphs, val_graphs):
        """
        Train the GCN model.

        Args:
            train_graphs: list of PyG Data objects (training set)
            val_graphs:   list of PyG Data objects (validation set)
        """
        self.model = GCNModel(
            in_channels=train_graphs[0].x.shape[1],
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Handle class imbalance using training set only
        if self.use_class_weight:
            y_train = torch.tensor([g.y.item() for g in train_graphs], dtype=torch.float32)
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        train_loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=self.batch_size, shuffle=False)

        best_val_auc = 0.0
        patience_counter = 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y.float().squeeze())
                loss.backward()
                optimizer.step()

            # Validation AUC every 10 epochs
            if epoch % 10 == 0:
                val_auc = self._evaluate_auc(val_loader)
                self.history.append((epoch, val_auc))
                print(f"Epoch {epoch:3d} | Val AUC: {val_auc:.4f}")

                # Save best checkpoint
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience // 10:
                        print(f"Early stopping at epoch {epoch} | Best Val AUC: {best_val_auc:.4f}")
                        break

        # Load best checkpoint
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()
        print(f"Training done. Best Val AUC: {best_val_auc:.4f}")

    def _evaluate_auc(self, loader):
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch.y.cpu().numpy())
        return roc_auc_score(all_labels, all_probs)

    def predict_proba(self, graphs):
        """
        Returns probability array of shape [N, 2].
        """
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out).cpu().numpy()
                all_probs.extend(probs)
        all_probs = np.array(all_probs)
        return np.stack([1 - all_probs, all_probs], axis=1)  # shape [N, 2]

    def predict(self, graphs):
        """
        Returns predicted labels of shape [N,].
        """
        proba = self.predict_proba(graphs)
        return (proba[:, 1] >= 0.5).astype(int)


