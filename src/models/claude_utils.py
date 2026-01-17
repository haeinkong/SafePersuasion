import anthropic
import os
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
from loader import load_prompt, load_dataset
import importlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_claude_answer(system_prompt, user_prompt):
    load_dotenv()
    client = anthropic.Anthropic(api_key = os.getenv("ANTHROPIC_API_KEY"))
    
    # print("System Prompt:", system_prompt)
    # print("User Prompt:", user_prompt)

    full_prompt = system_prompt + "\n\n" + user_prompt
    
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        temperature=0.1,
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    answer = response.content[0].text.strip()
    return answer

def run_claude(task, prompt_type):    
    df = load_dataset(task)
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
        desc=f"CLAUD | {task} | {prompt_type}"
        ):
        
        if prompt_type == "few-shot":
            fewshot_prompt = fewmod.make_fewshot_prompt(df, idx)
            system_prompt = base_prompt['system_prompt'] + " Do not include any explanation or reasoning in your answer." + "\n\n" + fewshot_prompt
            user_prompt = base_prompt['user_prompt'].format(text=row['text'])
            
        else:
            system_prompt = base_prompt['system_prompt'] + " Do not include any explanation or reasoning in your answer."
            user_prompt = base_prompt['user_prompt'].format(text=row['text'])
            
        answer = get_claude_answer(system_prompt, user_prompt)    
        preds.append(answer)
        
    df['llm_answer'] = preds
    
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{task}-claude-{prompt_type}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"All predictions saved to: {output_path}\n") 
