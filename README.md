# RSA-Control

This is the implementation of RSA-Control from "RSA-Control: A Pragmatics-Grounded Lightweight Controllable Text Generation Framework". The work will appear in the EMNLP, 2024.

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

### Citation
If you use this repository in your research, please cite:
```bibtex
@inproceedings{wang-demberg-2024-rsa,
    title = "{RSA}-Control: A Pragmatics-Grounded Lightweight Controllable Text Generation Framework",
    author = "Wang, Yifan  and
      Demberg, Vera",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.318/",
    doi = "10.18653/v1/2024.emnlp-main.318",
    pages = "5561--5582",
    abstract = "Despite significant advancements in natural language generation, controlling language models to produce texts with desired attributes remains a formidable challenge. In this work, we introduce RSA-Control, a training-free controllable text generation framework grounded in pragmatics. RSA-Control directs the generation process by recursively reasoning between imaginary speakers and listeners, enhancing the likelihood that target attributes are correctly interpreted by listeners amidst distractors. Additionally, we introduce a self-adjustable rationality parameter, which allows for automatic adjustment of control strength based on context. Our experiments, conducted with two task types and two types of language models, demonstrate that RSA-Control achieves strong attribute control while maintaining language fluency and content consistency. Our code is available at https://github.com/Ewanwong/RSA-Control."
}
