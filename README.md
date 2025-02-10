# DHd25_WwwwKlassifiziert
Data and Code related to the Poster "Warum wird was wie klassifiziert? Scalable Reading + Explainable AI am Beispiel historischer Lebensverläufe" presented at DHd 2025 illustrating a classification workflow that includes XAI.

# Requirements

- all scripts where developed and tested on Python 3.12.9
- install dependencies from `requirements.txt` (e.g. by using [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments))
- clone `Transformer-Explainability` from https://github.com/pdbro2k/Transformer-Explainability to `scripts/vendor`

## Sentiment Model

Choose one of the following three options:

- clone `LaTeCH-CLfL24_MoravianSentiment` from https://github.com/pdbro2k/LaTeCH-CLfL24_MoravianSentiment
    - execute `scripts/fine-tuning/fine-tune_gbert.ipynb` to fine-tune `deepset/gbert-base` on historical biographies and build the model `gbert-base-moravian-sentiment`
    - save `gbert-base-moravian-sentiment` to `models/sentiment/fine-tuned`
- save a different BERT model to `models/sentiment/fine-tuned` and change the value of `CLASSIFIER_PATH` at the beginning of `scripts/main.ipynb`
- change the value of `CLASSIFIER_PATH` at the beginning of `scripts/main.ipynb` to one of a sentiment model from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-classification&language=de) (c.f. `scripts/variation_example.ipynb`)

# Dir Structure

```bash
DHd25_WwwwKlassifiziert
│
├───data
│   ├───sentence-tokenized # The raw corpus split into sentences stored as a CSV file with the columns "source" (the source text ID), "n" (a running number per source text) and "text" (the sentence)
│   └───sentiment
│       ├───predictions # The tokenized corpus annotated with an additional "label" column  
│       └───predictions_explained # Sample HTML tables for annotated single texts with the columns "source", "n", "text" (the sentence with the impact of each subword token on the final classification highlighted), "label" and one additional column per label with the respective probability
├───models
│   └───sentiment
│       └───fine-tuned # Placeholder dir for a fine-tuned model
└───scripts
    ├───modules
    │   ├───aggregation.py # Contains a class for aggregating annotations of single texts to (sub)corpora and plotting these aggregations 
    │   ├───classification.py # Contains a class and helper functions for working with sentiment transformer models
    │   ├───explanation.py # Contains a class for explaining the classification of a BERT model
    │   └───segmentation.py # Contains a class for segmenting texts into steps of equal length
    ├───vendor # Placeholder dir for external modules
    ├───main.ipynb # Main Jupyter notebook used to execute the proposed workflow
    └───variation_example.ipynb # Additional Jupyter notebook illustrating customization options
```

# Data

The Data is derived from a 18th/19th century biographical subcorpus of the [ADB](https://www.deutsche-biographie.de/). Every original biography is licenced as [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) and can be retrieved with a URL following the pattern `https://www.deutsche-biographie.de/{SOURCE}.html#adbcontent`.