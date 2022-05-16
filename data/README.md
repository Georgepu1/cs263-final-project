To get the datasets (CoLA and SST-2), follow the next steps exactly for reproducibility.
- 1) CoLA: go on https://nyu-mll.github.io/CoLA/ and download v1.1 and store the entire "cola_public" directory (unchanged) into this data folder (included).
    - CoLA Resources: https://paperswithcode.com/sota/linguistic-acceptability-on-cola and https://nyu-mll.github.io/CoLA/

- 2) Stanford Sentiment Treebank: Instead of using the treebank structure for sentiment classification, we will use the HuggingFace SST dataset and transform it into a binary sentiment classification task by rounding each label to 0 or 1. The steps of installing are as follows: "pip install datasets" then inside a python script, "from datasets import load_dataset" and "dataset = load_dataset("sst", "default")"
    - SST Resources: https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary and https://huggingface.co/datasets/sst and https://huggingface.co/docs/datasets/installation