"""
Fingerprint-based ML models for BBB permeability prediction.
Models:
    - LogisticRegressionBBB  (Logistic Regression + GridSearchCV over C)
    - SVMBBB                 (SVM RBF kernel + grid over C, gamma)
    - MLPBBB                 (PyTorch MLP: 2048->256->128->64->1)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR,
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_PATIENCE
)


#  Logistic Regression
class LogisticRegressionBBB:
    """Logistic Regression on ECFP4 fingerprints with hyperparameter tuning.

    Searches over C = [0.001, 0.01, 0.1, 1, 10, 100] and selects the
    value that maximises ROC-AUC on the validation set.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.scaler = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)

        best_auc, best_model = -1, None
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            lr = LogisticRegression(
                C=C, random_state=self.seed, max_iter=1000, solver='lbfgs',
                class_weight='balanced'
            )
            lr.fit(X_train_sc, y_train)
            val_proba = lr.predict_proba(X_val_sc)[:, 1]
            auc = roc_auc_score(y_val, val_proba)
            if auc > best_auc:
                best_auc = auc
                best_model = lr

        self.model = best_model

    def predict(self, X) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc)

    def predict_proba(self, X) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        return self.model.predict_proba(X_sc)


#  SVM (RBF kernel)
class SVMBBB:
    """SVM with RBF kernel on ECFP4 fingerprints.

    Searches over C = [0.1, 1, 10] and gamma = ['scale', 'auto'].
    If RBF training exceeds ~15 min on real data, switch to
    kernel='linear' and document the change.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.scaler = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)

        best_auc, best_model = -1, None
        for C in [0.1, 1, 10]:
            for gamma in ['scale', 'auto']:
                svm = SVC(
                    kernel='rbf', C=C, gamma=gamma,
                    probability=True, random_state=self.seed,
                    class_weight='balanced'
                )
                svm.fit(X_train_sc, y_train)
                val_proba = svm.predict_proba(X_val_sc)[:, 1]
                auc = roc_auc_score(y_val, val_proba)
                if auc > best_auc:
                    best_auc = auc
                    best_model = svm

        self.model = best_model

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        return self.model.predict_proba(X_sc)


#  MLP (PyTorch)
class _MLPNet(nn.Module):
    """Feed-forward net: 2048 -> 256 -> 128 -> 64 -> 1
    Each hidden layer: Linear -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(self, input_dim=2048, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPBBB:
    """PyTorch MLP wrapped in the same sklearn-style API.

    Architecture:  2048 -> 256 -> 128 -> 64 -> 1
    Training:      Adam + BCEWithLogitsLoss + early stopping on val AUC
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.scaler = None
        self.net = None
        self.device = torch.device('cpu')

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, X_train, y_train, X_val, y_val):
        self._set_seed()
        self.scaler = StandardScaler()
        X_tr = self.scaler.fit_transform(X_train).astype(np.float32)
        X_v  = self.scaler.transform(X_val).astype(np.float32)

        train_ds = TensorDataset(
            torch.tensor(X_tr), torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True
        )

        self.net = _MLPNet(
            input_dim=X_tr.shape[1],
            hidden_dims=MLP_HIDDEN_DIMS,
            dropout=MLP_DROPOUT,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=MLP_LR)
        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc, patience_ctr = -1, 0
        best_state = None

        for epoch in range(MLP_EPOCHS):
            # ── train
            self.net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()

            # ── validate
            val_proba = self._predict_proba_raw(X_v)
            auc = roc_auc_score(y_val, val_proba)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= MLP_PATIENCE:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    # ── internal helper ──
    def _predict_proba_raw(self, X_scaled: np.ndarray) -> np.ndarray:
        """P(positive) as 1-D array. Input must already be scaled."""
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self.net(xt).squeeze()
            return torch.sigmoid(logits).cpu().numpy()

    # ── public API (matches LR / SVM interface) ──
    def predict(self, X) -> np.ndarray:
        X_sc = self.scaler.transform(X).astype(np.float32)
        proba = self._predict_proba_raw(X_sc)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        X_sc = self.scaler.transform(X).astype(np.float32)
        p_pos = self._predict_proba_raw(X_sc)
        return np.column_stack([1 - p_pos, p_pos])
