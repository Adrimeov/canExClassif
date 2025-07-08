from pathlib import Path
import random
import numpy as np
import torch
import os

PROJECT_ROOT_DIR: Path = Path(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Full dataset paths
DATA_DIR = PROJECT_ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
NORMALIZED_DATA_DIR = DATA_DIR / "normalized"
CURATED_DATA_DIR = DATA_DIR / "curated"
RAW_FILES = [RAW_DATA_DIR / "GS010012.mp4", RAW_DATA_DIR / "GS010016.mp4"]
LSTUDIO_ANNOTATIONS = CURATED_DATA_DIR / "lstudio_annotations.json"

# Test dataset paths
TEST_DIR = PROJECT_ROOT_DIR / "test"
TEST_DATA_DIR = TEST_DIR / "data"
TEST_CSV_ANNOTATIONS = TEST_DATA_DIR / "annotations.csv"
TEST_RAW_FILES = [RAW_DATA_DIR / "sampled.mp4"]

# Numerical constants
FINAL_IMAGE_SIZE_W = 56
FINAL_IMAGE_SIZE_H = 24

# Torch seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# DF Schemas
PATH_COL = "path"
LABEL_COL = "label"
