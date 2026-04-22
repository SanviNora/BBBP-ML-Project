import os

# ── Paths
DATA_DIR          = 'data/'
RAW_DATA_PATH     = 'data/BBBP.csv'
PROCESSED_DIR     = 'data/processed/'
RESULTS_DIR       = 'results/'
CHECKPOINT_DIR    = 'checkpoints/'

# ── Reproducibility
SEEDS             = [42, 123, 7]
SEED              = 42

# ── Data
SMILES_COL        = 'smiles'
LABEL_COL         = 'p_np'
ECFP_RADIUS       = 2
ECFP_NBITS        = 2048
RANDOM_SPLIT      = (0.8, 0.1, 0.1)

# ── Evaluation
SIGNIFICANCE_LEVEL = 0.05

# ── MLP
MLP_HIDDEN_DIMS   = [256, 128, 64]
MLP_DROPOUT       = 0.3
MLP_LR            = 1e-3
MLP_EPOCHS        = 100
MLP_BATCH_SIZE    = 64
MLP_PATIENCE      = 20

# ── GCN
GCN_HIDDEN_DIM    = 128
GCN_NUM_LAYERS    = 3
GCN_DROPOUT       = 0.3
GCN_LR            = 1e-3
GCN_EPOCHS        = 150
GCN_BATCH_SIZE    = 64
GCN_PATIENCE      = 20
