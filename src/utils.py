import json
import logging
import logging.config
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

def setup_logger(name=None, stdout: bool=True, file=None, level="INFO", mode="detail"):
    """Function to setup logging configuration"""
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)

    if mode == "basic":
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] %(message)s')
    
    if file:
        # Create a file handler
        file = Path(file)
        os.makedirs(file.parent, exist_ok=True)
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout:
        # Create a stream handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def prepare_file(path) -> None:
    file_path = Path(path) 
    os.makedirs(file_path.parent, exist_ok=True)

def read_jsonl(path: str, n: int = 0) -> list[dict]:
    data = []
    if n <= 0:
        data = [json.loads(line) for line in open(path).readlines() if line.strip() != ""]
    else:
        with open(path) as f:
            for line in f:
                if line.strip() != "":
                    data.append(json.loads(line.strip()))
                    if len(data) >= n:
                        break
    logger.info(f"Read {len(data)} lines from `{path}`")
    return data

def read_jsonl_chunk(path: str, chunk_size=10000, n: int = 0) -> Generator[list[dict], Any, Any]:
    data = []
    cnt = 0
    with open(path) as f:
        for line in tqdm(f, desc=path, ncols=0):
            if line.strip() != "":
                data.append(json.loads(line.strip()))
                cnt += 1
                if n > 0 and cnt >= n:
                    break
                if len(data) >= chunk_size:
                    yield data
                    data = []
        if data:
            yield data

def write_jsonl(data: list[dict], path: str, mode: str = 'w', verbose=True) -> None:
    prepare_file(path)
    with open(path, mode) as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')
    if verbose:
        logger.info(f"Saved {len(data)} lines to `{path}`") 

def read_pickle(path, verbose=True):
    if verbose:
        logger.info(f"Load data from `{path}`")
    return pickle.load(open(path, 'rb'))

def write_pickle(data, path, verbose=True):
    if verbose:
        logger.info(f"Save data to `{path}`")
    prepare_file(path)
    pickle.dump(data, open(path, 'wb'))

def get_files_of_type(input_dir, file_type='jsonl'):
    ext = "." + file_type
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(ext)]
    files.sort()
    logger.info(f"List {len(files)} files in `{input_dir}`")
    return files 

def dict_to_text(cfg: dict, indent: int = 2, sort: bool = False) -> str:
    return yaml.dump(cfg, indent=indent, default_flow_style=False, sort_keys=sort)

def current_time(format="%Y-%m-%d_%H-%M-%S"):
    """Return formatted date time string"""
    return datetime.now().strftime(format)

def set_seed(seed=42):
    import random
    import numpy as np
    import torch 

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    print(current_time()) 
