import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model_file_path:str, state_dict:dict, args:list, eval:float, step:int, silent:bool=False):
    checkpoint = dict()
    if not silent:
        print("Saving model...")
    checkpoint["state_dict"] = state_dict
    checkpoint["args"] = args
    checkpoint["eval"] = eval
    checkpoint["step"] = step
    torch.save(checkpoint, model_file_path)
    if not silent:
        print("Model saved:", step, eval)


def load_model(model_file_path:str, create_func:callable, device:str="cpu", silent:bool=False):
    if not silent:
        print("Loading model...")
    checkpoint = torch.load(model_file_path, weights_only=False)
    state_dict = checkpoint["state_dict"]
    args = checkpoint["args"]
    eval = checkpoint["eval"]
    step = checkpoint["step"]
    model = create_func(*args).to(device)
    model.load_state_dict(state_dict)
    if not silent:
        print("Model loaded:", step, eval)
    return model
