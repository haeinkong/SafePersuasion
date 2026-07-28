import anthropic
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
from runner import run_inference


def get_claude_answer(client, system_prompt, user_prompt):
    full_prompt = system_prompt + "\n\n" + user_prompt

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        temperature=0.1,
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.content[0].text.strip()


def run_claude(task, prompt_type):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    get_answer = lambda system_prompt, user_prompt: get_claude_answer(client, system_prompt, user_prompt)
    run_inference(
        "claude", "CLAUD", get_answer, task, prompt_type,
        system_suffix=" Do not include any explanation or reasoning in your answer."
    )
