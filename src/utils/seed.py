import importlib.util
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # torch is optional, so check before importing and this starts working the moment a project installs it
    if importlib.util.find_spec("torch"):
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
