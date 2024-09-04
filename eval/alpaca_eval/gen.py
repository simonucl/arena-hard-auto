import os
import json
import argparse
import logging
import random
import torch
import datasets
import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    prompts = []
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        prompts.append(prompt)

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine

    if args.model_name_or_path is not None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer
        )

        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted_prompt)

        outputs = []
        for prompt in formatted_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(decoded_output[len(prompt):].strip())

        if "llama-3.1" in args.model_name_or_path.lower() or "llama-3" in args.model_name_or_path.lower():
            stop_tokens = tokenizer.convert_tokens_to_ids(["</s>", "<|im_end|>", "<|endoftext|>"])
            outputs = [output.split(tokenizer.decode(stop_tokens[0]))[0] for output in outputs]

    model_results = [{**example, 'output': output, 'generator': model_name} for example, output in zip(alpaca_eval_data, outputs)]
    with open(os.path.join(args.save_dir, f"{model_name}-greedy-long-output.json"), "w") as fout:
        json.dump(model_results, fout, indent=2)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to the reference outputs. "
             "Alpaca_eval leaderboard use text-davinci-003 to generate the reference outputs, "
             "but they limit the max_tokens to 300, which is a bit unfair for text-davinci-003. "
             "Here we keep this default setup to make numbers comparable to their leaderboard. "
             "But you can also use the regenerated reference outputs with max_tokens=2048 "
             "hosted at https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token.",
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--stop_id_sequences",
        type=str,
        default=None,
        help="The stop id sequences for the model.",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)