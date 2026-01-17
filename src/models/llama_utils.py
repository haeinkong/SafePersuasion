import importlib
import torch
from tqdm import tqdm
import logging
import transformers
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import warnings
warnings.filterwarnings('ignore')
from loader import load_prompt, load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_llama_answer(system_prompt, user_prompt):
       
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        )
    
    # print("System Prompt:", system_prompt)
    # print("User Prompt:", user_prompt)
    
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        temperature=0.1
    )
    
    answer = outputs[0]["generated_text"][-1]["content"]
   
    return answer

def run_llama(task, prompt_type):
    df = load_dataset(task).head(3)
    
    print(f"Loaded {len(df)} samples")

    # Zero-shot & COT
    base_prompt = None
    if prompt_type != "few-shot":
        base_prompt = load_prompt(task, prompt_type)

    # Few-shot
    if prompt_type == "few-shot":
        base_prompt = load_prompt(task, "zero-shot")
        module_path = f"prompts.{task}.fewshot_utils"
        fewmod = importlib.import_module(module_path)

    preds = []
               
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"LLAMA | {task} | {prompt_type}"
        ):
        
        if prompt_type == "few-shot":
            fewshot_prompt = fewmod.make_fewshot_prompt(df, idx)
            system_prompt = base_prompt['system_prompt'] + "\n\n" + fewshot_prompt
            user_prompt = base_prompt['user_prompt'].format(text=row['text'])
            
        else:
            system_prompt = base_prompt['system_prompt']
            user_prompt = base_prompt['user_prompt'].format(text=row['text'])
            
        answer = get_llama_answer(system_prompt, user_prompt)    
        preds.append(answer)
        
    df['llm_answer'] = preds
    
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{task}-llama-{prompt_type}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"All predictions saved to: {output_path}\n") 

