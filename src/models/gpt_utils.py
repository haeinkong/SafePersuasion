from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
from runner import run_inference


def get_gpt_answer(client, system_prompt, user_prompt):
    response = client.responses.create(
        model="gpt-4.1-2025-04-14",
        instructions=system_prompt,
        input=user_prompt,
        temperature=0.1
    )
    return response.output_text


def run_gpt(task, prompt_type):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    get_answer = lambda system_prompt, user_prompt: get_gpt_answer(client, system_prompt, user_prompt)
    run_inference("gpt", "GPT", get_answer, task, prompt_type)
