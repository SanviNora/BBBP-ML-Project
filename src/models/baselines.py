import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from config import SEED


class LipinskiClassifier:
    """Rule-based BBB predictor using Lipinski's Rule of Five."""

    def _satisfies_lipinski(self, smiles: str) -> int:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        mw  = Descriptors.MolWt(mol)
        lp  = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        return int(mw < 500 and lp < 5 and hbd < 5 and hba < 10)

    def predict(self, smiles_list: list) -> np.ndarray:
        return np.array([self._satisfies_lipinski(s) for s in smiles_list])

    def predict_proba(self, smiles_list: list) -> np.ndarray:
        p = self.predict(smiles_list).astype(float)
        return np.stack([1-p, p], axis=1)

    def get_property_breakdown(self, smiles_list: list) -> pd.DataFrame:
        rows = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                rows.append({'smiles': s, 'mw': None, 'logp': None, 'hbd': None, 'hba': None})
                continue
            rows.append({'smiles': s,
                         'mw':   Descriptors.MolWt(mol),
                         'logp': Descriptors.MolLogP(mol),
                         'hbd':  Descriptors.NumHDonors(mol),
                         'hba':  Descriptors.NumHAcceptors(mol)})
        return pd.DataFrame(rows)


class BBBRandomForest:
    """Random Forest on ECFP4 fingerprints with light hyperparameter search."""

    def __init__(self, seed: int = SEED):
        self.seed  = seed
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        best_auc, best_model = 0, None
        for n_est in [100, 300]:
            for max_d in [None, 10, 20]:
                m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced',
                                           random_state=self.seed, n_jobs=-1)
                m.fit(X_train, y_train)
                auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
                if auc > best_auc:
                    best_auc, best_model = auc, m
        self.model = best_model
        return self

    def predict(self, X):       return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
