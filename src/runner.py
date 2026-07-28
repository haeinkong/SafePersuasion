import os
from tqdm import tqdm
from loader import load_prompt, load_dataset
from prompts.fewshot_utils import make_fewshot_prompt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label column, few-shot samples per class) per task
FEWSHOT_CONFIG = {
    "binary": ("first_label", 2),
    "multi-m": ("second_label", 1),
    "multi-rp": ("second_label", 1),
}


def run_inference(model_name, desc_label, get_answer, task, prompt_type, sample_size=None, system_suffix=""):
    df = load_dataset(task)
    if sample_size is not None:
        df = df.head(sample_size)

    print(f"Loaded {len(df)} samples")

    base_prompt = load_prompt(task, "zero-shot" if prompt_type == "few-shot" else prompt_type)
    label_col, n_samples = FEWSHOT_CONFIG[task] if prompt_type == "few-shot" else (None, None)

    preds = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{desc_label} | {task} | {prompt_type}"):
        system_prompt = base_prompt["system_prompt"] + system_suffix
        if prompt_type == "few-shot":
            system_prompt += "\n\n" + make_fewshot_prompt(df, idx, label_col, n_samples)
        user_prompt = base_prompt["user_prompt"].format(text=row["text"])

        preds.append(get_answer(system_prompt, user_prompt))

    df["llm_answer"] = preds

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{task}-{model_name}-{prompt_type}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"All predictions saved to: {output_path}\n")
