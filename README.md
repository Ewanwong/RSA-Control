# RSA-Control

This is the implementation of RSA-Control from "A Pragmatics-Grounded Lightweight Controllable Text Generation Framework". 

## Environment

Our codes require `transformers==4.38.2`.

## Run

To reproduce results from the paper, please run:

### Toxicity Reduction
`python toxicity_bias_mitigation/detoxification.py --output_dir detoxification_results --api_key YOUR_PERSPECTIVE_API_KEY --adjustable --two_models`

### Bias Mitigation
`python toxicity_bias_mitigation/crows_pairs_experiments.py --output_dir debiasing_results  --adjustable --two_models`

### Readability-Controlled Summarization
`python readability_controlled_summarization/readability_controlled_summarization.py --output_dir readability_control_results --adjustable`

CrowS-Pairs and RealToxicityPrompts datasets for toxicity and bias reduction experiments can be found in `toxicity_bias_mitigation/data/`, and we use CNN/DM from huggingface transformers.

