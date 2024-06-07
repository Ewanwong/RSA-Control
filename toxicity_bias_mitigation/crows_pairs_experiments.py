import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict
from transformers import AutoConfig
import torch
from tqdm import tqdm
from PragmaticGPT2 import PragmaticGPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from perspective_api import PerspectiveApiScorer
from io_utils import load_prompts, ModelOutput, Prompt

from crows_pairs import CrowSPairsRunner


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="toxicity_bias_mitigation/data/crows_pairs_anonymized.csv", help="Path to the input file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to a directory into which generated outputs are written")
    parser.add_argument("--models", type=str, nargs='+', default=['gpt2-large'],
                        help="The specific models to run self-debiasing experiments for (e.g., 'gpt2-medium gpt2-large')")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Value for the rational parameter")
    parser.add_argument("--beta", type=float, default=10.0,)
    parser.add_argument("--adjustable", action='store_true',)
    parser.add_argument("--two_models", action='store_true',)
    parser.add_argument("--classifier_model", default='gpt2', type=str,)
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for initializing the random number generator used for sampling")
    parser.add_argument("--modes", type=str, default=['debiased'],)

    args = parser.parse_args()
    print(f"Parameters: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'config.txt'), 'a', encoding='utf8') as fh:
        fh.write(f'Parameters: {args}\n')
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for model_name in args.models:
        if 'default' in args.modes:
            with open(os.path.join(args.output_dir, f'{model_name.replace("/", "_")}.txt'), 'a', encoding='utf8') as fh:

                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                runner = CrowSPairsRunner(model, tokenizer, input_file=args.input_file)
                all_trials = {bias_type: [] for bias_type in runner.bias_types}
                for bias_type in runner.bias_types:                   
                    results = runner.run(bias_type)
                    print(f'default {model_name} {bias_type}: {results}')
                    fh.write(f'default {model_name} {bias_type}: {results}\n')

        if 'debiased' in args.modes:
            print(f"Running experiments for {model_name} with debiased")
            with open(os.path.join(args.output_dir, f'{model_name.replace("/", "_")}.txt'), 'a', encoding='utf8') as fh:
                if args.two_models:
                    model = PragmaticGPT2LMHeadModel(args.classifier_model, model_name, alpha=args.alpha, beta=args.beta, num_classes=2, adjustable=args.adjustable).to(device)
                    if hasattr(model, 'classifier_model'):
                        model.classifier_model.to(device)
                else:
                    model = PragmaticGPT2LMHeadModel(model_name, alpha=args.alpha, beta=args.beta, num_classes=2, adjustable=args.adjustable).to(device)
                #torch.set_default_dtype(torch.float32)
                runner = CrowSPairsRunner(model, model.tokenizer_right, input_file=args.input_file)
                for bias_type in runner.bias_types:                    
                    results = runner.run(bias_type)
                    print(f'debiased {model_name} {bias_type}: {results}')
                    fh.write(f'debiased {model_name} {bias_type}: {results}\n')
        with open(os.path.join(args.output_dir, f'{model_name.replace("/", "_")}.txt'), 'a', encoding='utf8') as fh:
            fh.write("===============================================================================================\n")
        