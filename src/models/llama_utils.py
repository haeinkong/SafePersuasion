import torch
import transformers
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
from runner import run_inference


def get_llama_answer(pipeline, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        temperature=0.1
    )

    return outputs[0]["generated_text"][-1]["content"]


def run_llama(task, prompt_type):
    # load the model once and reuse it across all samples
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )
    get_answer = lambda system_prompt, user_prompt: get_llama_answer(pipeline, system_prompt, user_prompt)
    run_inference("llama", "LLAMA", get_answer, task, prompt_type, sample_size=3)
