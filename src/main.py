import argparse
from models.llama_utils import run_llama
from models.gpt_utils import run_gpt
from models.claude_utils import run_claude

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "gpt", "claude"], required=True)
    parser.add_argument("--task", choices=["binary", "multi-m", "multi-rp"], required=True)
    parser.add_argument("--prompt", choices=["zero-shot", "few-shot", "chain-of-thought"], required=True)
    args = parser.parse_args()

    print(f"\nRunning {args.model.upper()} on {args.task} task with {args.prompt} prompt")

    if args.model == "llama":
        run_llama(args.task, args.prompt)
    elif args.model == "gpt":
        run_gpt(args.task, args.prompt)
    elif args.model == "claude":
        run_claude(args.task, args.prompt)

if __name__ == "__main__":
    main()
