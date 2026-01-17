import os
import json
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

def load_dataset(task):
    """
    Load dataset for the given task name.
    Uses explicit mapping to dataset filenames.
    Works regardless of working directory.
    """
    dataset_map = {
        "binary": "SafePersuasion.csv",
        "multi-m": "SafePersuasion_Multi_M.csv",
        "multi-rp": "SafePersuasion_Multi_RP.csv"
    }

    if task not in dataset_map:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(dataset_map.keys())}.")

    path = os.path.join(ROOT_DIR, "dataset", dataset_map[task])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"Loading dataset for {task} task → {os.path.basename(path)}")
    return pd.read_csv(path)

def load_prompt(task, prompt_type):

    path = os.path.join(ROOT_DIR, "src/prompts", task, f"{prompt_type}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)