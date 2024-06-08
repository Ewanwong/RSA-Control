from PragmaticLlama import PragmaticLlamaForCausalLM
import transformers
import torch
from transformers import AutoTokenizer, AutoConfig
import textstat
from datasets import load_dataset
import json
from tqdm import tqdm
import random
import numpy as np
import os 

def set_random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse

parser = argparse.ArgumentParser(description='Pragmatic Summarization')
parser.add_argument('--output_dir', type=str, required=True, help='output directory')
parser.add_argument('--alpha', type=float, default=10, help='alpha')
parser.add_argument('--beta', type=float, default=10, help='beta')
parser.add_argument('--adjustable', action='store_true', help='adjustable')
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument('--seed', type=int, default=42, help='seed')

args = parser.parse_args()
output_file = os.path.join(args.output_dir, "output.json")

# load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")['test']
dataset = [{"article": d['article'], "reference": d['highlights']} for d in dataset]
# dataset = dataset[:10]
# load model

model_name = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.alpha = args.alpha
config.beta = args.beta
config.adjustable = args.adjustable
config.seed = args.seed
config.num_beams=args.num_beams

set_random_seed(args.seed)

prompt = "Summarize the following news article in three sentences for a primary-school student: " # for a college professor
readable_prompt = "Write a story for a primary-school student"
unreadable_prompt = "Write a research paper abstract for a college professor"

prompts = [prompt, readable_prompt, unreadable_prompt] ##################
config.num_classes = len(prompts) - 1
config.prompts = prompts
config.save_pretrained(args.output_dir)

model = PragmaticLlamaForCausalLM(config=config)

output_dataset = []
# summarize

for i, data in tqdm(enumerate(dataset)):
    input_texts = [data["article"]] + [''] * config.num_classes
    
    outputs = model.debiased_generation(prompts=prompts, input_texts=input_texts, num_beams=args.num_beams, max_new_tokens=200)
    #data['default_prediction'] = data['prediction']
    data['id'] = i
    response = [r.split("[/INST]")[-1] for r in outputs]
    response = [r.strip().split("\n\n") for r in response]
    # if r consists of multiple parts, select the longest one
    response = [max(r, key=len) for r in response]
    print(response[0])
    print(textstat.flesch_reading_ease(response[0]))
    data["prediction"] = response[0]
    data['fre'] = textstat.flesch_reading_ease(response[0])
    output_dataset.append(data)

with open(output_file , "w") as f:
    json.dump(output_dataset, f, indent=4, ensure_ascii=False)
