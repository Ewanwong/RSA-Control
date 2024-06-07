import argparse
import torch
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from os import listdir
from os.path import isfile, join


def load_data(examples_filename):
    with open(examples_filename, 'r') as f:
        examples = f.readlines()
    continuation_examples = [json.loads(example)['continuations'][0]['text'] for example in examples]
    full_text_examples = [json.loads(example)['prompt'] + json.loads(example)['continuations'][0]['text'] for example in examples]
    prompt_text_examples = [json.loads(example)['prompt'] for example in examples]
    return continuation_examples, full_text_examples, prompt_text_examples

def compute_conditional_perplexity(prompt_examples, full_examples, model, tokenizer, device):
    perplexities = []
    for prompt_example, full_example in tqdm(zip(prompt_examples, full_examples)):
        prompt_input_ids = tokenizer([prompt_example], return_tensors='pt').to(device)
        prompt_outputs = model(**prompt_input_ids, labels=prompt_input_ids['input_ids'].clone())
        prompt_loss = prompt_outputs.loss.item()*(prompt_input_ids['input_ids'].shape[1]-1)
        
        full_input_ids = tokenizer([full_example], return_tensors='pt').to(device)
        full_outputs = model(**full_input_ids, labels=full_input_ids['input_ids'].clone())
        full_loss = full_outputs.loss.item()*(full_input_ids['input_ids'].shape[1]-1)

        conditional_loss = full_loss - prompt_loss
        perplexities.append(conditional_loss)

    return perplexities

def compute_perplexity(examples, model, tokenizer, device):
    perplexities = []
    for example in tqdm(examples):
        input_ids = tokenizer([example], return_tensors='pt').to(device)
        outputs = model(**input_ids, labels=input_ids['input_ids'].clone())
        loss = outputs.loss
        perplexities.append(loss.item()*(input_ids['input_ids'].shape[1]-1))
    return perplexities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_dir", type=str, required=True,
                        help="Path to a file to which the output of the self-diagnosis experiment is written")
    
    parser.add_argument("--perplexity_model", type=str, default='EleutherAI/gpt-j-6b', # EleutherAI/gpt-j-6b
                        help="The specific model to compute perplexity for (e.g., 'gpt2-medium')")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    # find all txt files in a directory
    example_files = [f for f in listdir(args.examples_dir) if isfile(join(args.examples_dir, f)) and f.endswith('.txt') and 'prompted_generations_' in f]
    output_files = [f.replace('prompted_generations_', 'conditional_perplexity_') for f in example_files]

    if len(example_files) != 0:

        # load model
        tokenizer = AutoTokenizer.from_pretrained(args.perplexity_model)
        model = AutoModelForCausalLM.from_pretrained(args.perplexity_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        torch.set_grad_enabled(False)
        model.to(device)

        for examples_file, output_file in zip(example_files, output_files):            
            if os.path.exists(join(args.examples_dir, output_file)):
                continue            
            print(examples_file)
            continuation_examples, full_text_examples, prompt_text_examples = load_data(join(args.examples_dir, examples_file))

            # compute perplexity
            conditional_perplexities = compute_conditional_perplexity(prompt_text_examples, full_text_examples, model, tokenizer, device)

            # write output to file
            with open(join(args.examples_dir, output_file), 'w') as f:
                f.write('Average conditional perplexity: ' + str(sum(conditional_perplexities)/len(conditional_perplexities)) + '\n')
                f.write('Conditional perplexities: ' + str(conditional_perplexities) + '\n')