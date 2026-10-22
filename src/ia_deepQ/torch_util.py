import random

import numpy as np
import torch

from . import util as utl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(state_dict, args, eval, step, prefix="", timestamp_suffix=True, silent=False):
    checkpoint = dict()
    if not silent:
        print("Saving model...")
    checkpoint["state_dict"] = state_dict
    checkpoint["args"] = args
    checkpoint["eval"] = eval
    checkpoint["step"] = step
    suffix = utl.get_timestamp() if timestamp_suffix else ""
    model_file_path = prefix + suffix + ".pt"
    torch.save(checkpoint, model_file_path)
    if not silent:
        print("Model saved:", step, eval)


def load_model(model_file_path, create_func, device, silent=False):
    if not silent:
        print("Loading model...")
    checkpoint = torch.load(model_file_path)
    state_dict = checkpoint["state_dict"]
    args = checkpoint["args"]
    eval = checkpoint["eval"]
    step = checkpoint["step"]
    model = create_func(*args).to(device)
    model.load_state_dict(state_dict)
    if not silent:
        print("Model loaded:", step, eval)
    return model

