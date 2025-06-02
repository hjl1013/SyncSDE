import sys 
import os 
import numpy as np 
import torch
import argparse

from syncsde.pipelines.pipeline_mesh_texturing import MeshTextureModel
from syncsde.synctweedies.config.mesh_config import load_mesh_config

def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything()

    config = load_mesh_config()
    model = MeshTextureModel(config)
    
    with open(config.prompt_path, "r") as f:
        prompt = f.read()
    
    seed = config.random_seed

    model(prompt, seed)