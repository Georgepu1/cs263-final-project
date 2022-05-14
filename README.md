# cs263-final-project

- We perform multitask learning on classification NLP tasks and measure robustness across several adversarial attacks (char/word-level) and metrics. We will analyze this (hard-parameter sharing) across model architectures: word embeddings + LSTM, BERT + classification heads, and "large" language models (BART) with prompting. After doing so, we will improve the robustness through smoothing and fine-tuning/adversarial training our models with adversarial cases and data augmentation. 
- Find relevant literature and codebases on multi-task learning/model architectures/model robustness techniques/adversarial attacks/datasets
- Datasets: CoLA (grammar acceptability) and SST-2 (sentiment)
- Hard Parameter sharing: Zip dataset to dataloader for each batch, randomly sample from d1, d2, ...; calculate loss one by one and add together to get overall loss
- Prompt with BART, GPT-2 (not recommended XLNet), GPT-2 serve based-size model (12 layer transformers)
- Note: Data augmentation less feasible, grab paper for general ideas (e.g. working on parameters of model or function defined by model s' where small change in input won't have large effects)
- Lipschitz smoothing (enhance model by smoothing landscape): make transformations induced by model distort features drastically (take input and calculate gradient of current model w.r.t. input to tell you second order info of points (embedding) so sample around input point to see what does it look like => norm of gradient is differentiable to calculate the loss. 
- Note: char level modification would make token out of the range (BPE with BERT/GPT-2), maybe not only one token affected but multiple (e.g. bigly has 2 tokens => bigily will still be two tokens for cases like really; adding a dash between bigly will add one character but it would affect embedding space drastically). 
- pseudocode in pseudocode.py
