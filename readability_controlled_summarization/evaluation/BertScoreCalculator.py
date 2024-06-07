import json
import os
import torch
import numpy as np
from bert_score import score
from tqdm import tqdm

class BERTScoreCalculator:

    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def compute_bert_score(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        predictions = [pair['prediction'].strip() for pair in input_data]
        references = [pair['reference'].strip() for pair in input_data]
        results = score(predictions, references, lang="en", model_type='roberta-large', device=self.device)
        results = [{'precision': round(P.item() * 100, 4), 'recall': round(R.item() * 100, 4), 'f1': round(F1.item() * 100, 4)} for P, R, F1 in zip(*results)]

        average_scores = {k: round(np.mean([res[k] for res in results]), 4) for k in results[0]}
        results.append({'corpus_level': average_scores})
        return results

    def compute_bert_score_from_file(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_bert_score(data)

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_path', required=True, type=str, help='Path to the input JSON file')
    args = parser.parse_args()
    bert_score_calculator = BERTScoreCalculator()

    output_json_path = args.input_json_path.replace(".json", "_bertscore.json")
    bert_score_calculator.compute_bert_score_from_file(args.input_json_path, output_json_path)
    