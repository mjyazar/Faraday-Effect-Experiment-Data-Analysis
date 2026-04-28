import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
# Defines the directory where data files are stored.
DATA_DIR = BASE_DIR / "data"

# Load the dataset
data = pd.read_csv('')