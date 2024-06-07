import json
import os
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from datasets import load_metric
from tqdm import tqdm

# Download the Punkt tokenizer model
nltk.download('punkt')

class RougeCalculator:
    def __init__(self):
        # Removed spacy's model initialization
        self.rouge = load_metric('rouge')

    def add_newlines_to_sentences(self, text):
        # Use nltk's sentence tokenizer
        sentences = sent_tokenize(text)
        return '\n'.join(sentences)

    def compute_rouge(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        for pair in tqdm(input_data):
            try:
                reference = pair['reference'].strip()
                prediction = pair['prediction'].strip()
            except KeyError:
                raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
                
            reference = self.add_newlines_to_sentences(reference)
            prediction = self.add_newlines_to_sentences(prediction)
            scores = self.rouge.compute(predictions=[prediction], references=[reference], use_stemmer=True)

            # Round the results to 4 decimal places
            scores = {k: round(v.mid.fmeasure * 100, 4) for k, v in scores.items()}
            results.append(scores)

        average_scores = {k: round(np.mean([res[k] for res in results]), 4) for k in results[0]}
        results.append({'corpus_level': average_scores})
        return results

    def compute_rouge_from_file(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_rouge(data)

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_path', required=True, type=str, help='Path to the input JSON file')
    args = parser.parse_args()
    rouge_calculator = RougeCalculator()
    output_json_path = args.input_json_path.replace(".json", "_rouge.json")
    rouge_calculator.compute_rouge_from_file(args.input_json_path, output_json_path)
