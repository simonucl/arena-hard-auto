import os
import json
import argparse
import logging
import random
import torch
import datasets
import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
        
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

def load_hf_lm(
        model_name_or_path,
        revision=None,
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        gptq_model=False,
        token=os.getenv("HF_TOKEN", None),
    ):

    # Loading OLMo models from HF requires `trust_remote_code=True`.
    # TODO: Implement this via command-line flag rather than hardcoded list.
    trusted_models = ["allenai/OLMo-7B", "allenai/OLMo-7B-Twin-2T", "allenai/OLMo-1B"]
    if model_name_or_path in trusted_models:
        trust_remote_code = True
    else:
        trust_remote_code = False

    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True, trust_remote_code=trust_remote_code
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            revision=revision,
            device_map=device_map, 
            load_in_8bit=True,
            token=token,
            trust_remote_code=trust_remote_code
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                revision=revision,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                revision=revision,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()
    return model

def load_hf_tokenizer(
        model_name_or_path, 
        revision=None,
        tokenizer_name_or_path=None, 
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
    ):
        from transformers import AutoTokenizer
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token, revision=revision)
        except:
            # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token, revision=revision)
        # set padding side to left for batch generation
        tokenizer.padding_side = padding_side
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        revision=None,
        tokenizer_name_or_path=None,
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        gptq_model=False,
        padding_side="left",
        use_fast_tokenizer=True,
        token=os.getenv("HF_TOKEN", None),
    ):
        tokenizer = load_hf_tokenizer(
            model_name_or_path=model_name_or_path,
            revision=revision,
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_fast_tokenizer=use_fast_tokenizer,
            padding_side=padding_side,
            token=token,
        )
        model = load_hf_lm(
            model_name_or_path=model_name_or_path,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            convert_to_half=convert_to_half,
            gptq_model=gptq_model,
            token=token,
        )
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        return model, tokenizer

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    prompts = []
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        prompts.append(prompt)

    # <|end_of_text|>, <|eom_id|>, <|eot_id|>
    additional_stop_sequence = [["<|end_of_text|>"], ["<|eom_id|>"], ["<|eot_id|>"]]
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine

    if args.model_name_or_path is not None:
        # we always load the tokenizer for vllm or hf models
        tokenizer = load_hf_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                use_fast_tokenizer=not args.use_slow_tokenizer,
                revision=args.hf_revision,
            )

        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                tokenizer_revision=args.hf_revision,
                revision=args.hf_revision,
            )
            
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=args.max_new_tokens,
                stop=additional_stop_sequence,
            )
            # apply chat formatting
            formatted_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenizer=tokenizer, add_bos=False)
                formatted_prompts.append(formatted_prompt)
            prompts = formatted_prompts
                    
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
        else:
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                revision=args.hf_revision,
                load_in_8bit=args.load_in_8bit,
                device_map="auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

            # apply chat formatting
            formatted_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(formatted_prompt)
            prompts = formatted_prompts
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                stop_id_sequences=[tokenizer.convert_tokens_to_ids(stop) for stop in additional_stop_sequence],
            )
        # if "llama-3.1" in args.model_name_or_path.lower() or "llama-3" in args.model_name_or_path.lower():
        #     stop_tokens = tokenizer.convert_tokens_to_ids(["</s>", "<|im_end|>", "<|endoftext|>"])
        #     outputs = [output.split(tokenizer.decode(stop_tokens[0]))[0] for output in outputs]

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

    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vllm to generate the predictions.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
