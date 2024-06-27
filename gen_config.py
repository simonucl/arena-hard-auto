import yaml
from argparse import ArgumentParser
import os
def gen_api_config(model_name, model_path):
    api_config = {
        model_name: {
            "model_name": model_path,
            "endpoints": [
                {
                    "api_base": "http://0.0.0.0:8742/v1",
                    "api_key": "token-abc123",
                }
            ],
            "api_type": "openai",
            "parallel": 8,
    }
    }
    return api_config

def gen_answer_config(model_name):
    answer_config = {
        "bench_name": "arena-hard-v0.1",
        "temperature": 0.0,
        "max_tokens": 4096,
        "num_choices": 1,
        "model_list": [model_name],
    }
    return answer_config

def gen_judge_config(model_name):
    judge_config = {
        "bench_name": "arena-hard-v0.1",
        "judge_model": "gpt-4-1106-preview",
        "reference": False,
        "ref_model": "null",
        "baseline": True,
        "baseline_model": "gpt-4-0314",
        "pairwise": True,
        "temperature": 0,
        "max_tokens": 4096,
        "regex_pattern": r"\[\[([AB<>=]+)\]\]",
        "number_of_judgment_attempts": 2,
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\".",
        "prompt_template": ["<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"],
        "model_list": [model_name],
    }
    return judge_config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="meta-llama/Llama-3-Base-8B-SFT-SOAP"
    )
    args = parser.parse_args()
    
    # create directory under config
    model_name = os.path.basename(args.model_path)
    os.makedirs(f"config/{model_name}", exist_ok=True)
    api_config = gen_api_config(model_name, args.model_path)
    answer_config = gen_answer_config(model_name)
    judge_config = gen_judge_config(model_name)

    with open(f"config/{model_name}/api_config.yaml", "w") as fout:
        yaml.dump(api_config, fout, sort_keys=False)
    with open(f"config/{model_name}/gen_answer_config.yaml", "w") as fout:
        yaml.dump(answer_config, fout, sort_keys=False)
    with open(f"config/{model_name}/judge_config.yaml", "w") as fout:
        yaml.dump(judge_config, fout, sort_keys=False)

    