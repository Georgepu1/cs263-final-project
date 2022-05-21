import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import tqdm

from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers import AutoConfig, AutoModelForPreTraining
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from data import *
from evaluate import *
from model import *
from preprocess import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Modified LM-LSTM-CRF with Entity-Specific Models')
    parser.add_argument('--type', type=int, default="bert")

    args = parser.parse_args()

    # From the folder/pytorch_script.py
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    sst_train_data, sst_val_data, sst_test_data = get_sst_data()
    cola_train_data, cola_val_data, cola_test_data = get_cola_data(
        './data/cola_public/raw/')

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    max_epochs = 50

    if args.type == 'prompt':
        # TODO: implement both data and model
        train_set = GenDataset(tokenizer, max_length, train_set, train_replace_words, int(
            len(dataset) * .8), max_output_length)
        val_set = GenDataset(tokenizer, max_length, val_set, val_replace_words, int(
            len(dataset) * .1), max_output_length)
        test_set = GenDataset(tokenizer, max_length, test_set, test_replace_words, int(
            len(dataset) * .1), max_output_length)
        # train_batch_num = int(len(dataset) * .8) // train_batch_size + (int(len(dataset) * .8) % train_batch_size != 0)
        train_batch_num = len(train_set) // train_batch_size + \
            (len(train_set) % train_batch_size != 0)
        val_batch_num = len(val_set) // eval_batch_size + \
            (len(val_set) % eval_batch_size != 0)
        # test_batch_num = int(len(dataset) * .2) // eval_batch_size + (int(len(dataset) * .2) % eval_batch_size != 0)
        test_batch_num = len(test_set) // eval_batch_size + \
            (len(test_set) % eval_batch_size != 0)
        # initialize the model
        model = GenerativeModel(model_name, cache_dir, tokenizer)
        model.cuda(device=0)

        # optimizer
        param_groups = [{'params': model.parameters(), 'lr': learning_rate,
                        'weight_decay': weight_decay}]
        optimizer = AdamW(params=param_groups)
        schedule = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=train_batch_num*warmup_epoch,
                                                   num_training_steps=train_batch_num*max_epoch)

        summarizer_step = 0
        for epoch in range(1, max_epochs+1):

            # training
            model.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(tqdm(DataLoader(train_set, batch_size=train_batch_size // accumulate_step,
                                                              shuffle=True, drop_last=False, collate_fn=train_set.collate_fn))):
                loss = model(batch)
                loss = loss * (1 / accumulate_step)
                loss.backward()

                if (batch_idx + 1) % accumulate_step == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clipping)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()
            # validation
            model.eval()
            # track vals for entire sentence
            total_acc = 0
            total_perfect_match = 0
            total_N = 0
            total_mismatch = 0
            # vals for sentence excluding first word or when input has mismatch (e.g. first word skipped in output)
            skip_first_acc = 0
            skip_first_perfect_match = 0
            skip_first_N = 0
            skip_first_mismatch = 0

            for batch_idx, batch in enumerate(tqdm(DataLoader(val_set, batch_size=eval_batch_size,
                                                              shuffle=False, collate_fn=val_set.collate_fn))):

                pred_text = model.predict(
                    batch, max_length=max_output_length)  # takes a long time
                cur_acc, cur_perfect_match, cur_N, cur_mismatch = exact_match(
                    batch.target_text, pred_text)
                skip_acc, skip_perfect_match, skip_N, skip_mismatch = exact_match(batch.target_text,
                                                                                  pred_text, skip_first=True)
                total_acc += cur_acc
                skip_first_acc += skip_acc
                total_perfect_match += cur_perfect_match
                skip_first_perfect_match += skip_perfect_match
                total_N += cur_N
                skip_first_N += skip_N
                total_mismatch += cur_mismatch
                skip_first_mismatch += skip_mismatch
                # print("Current validation accuracy: {}, perfect matches: {}, N: {}, skipped outputs: {}".format(cur_acc/cur_N, cur_perfect_match, cur_N, cur_mismatch))

                print("Epoch: {}, Validation accuracy: {}, perfect matches: {}, N: {}, skipped outputs: {}".format(
                    epoch, total_acc/total_N, total_perfect_match, total_N, total_mismatch))
                print("Epoch: {}, Skip-first validation accuracy: {}, perfect matches: {}, N: {}, skipped outputs: {}".format(
                    epoch, skip_first_acc/skip_first_N, skip_first_perfect_match, skip_first_N, total_mismatch))

    else:
        # Generators
        training_set = MultitaskDataset(sst_train_data['sentence'], sst_train_data['label'],
                                        cola_train_data[0], cola_train_data[1])
        training_generator = torch.utils.data.DataLoader(
            training_set, **params)

        validation_set = MultitaskDataset(sst_val_data['sentence'], sst_val_data['label'],
                                          cola_val_data[0], cola_val_data[1])
        validation_generator = torch.utils.data.DataLoader(
            validation_set, **params)

        test_set = MultitaskDataset(sst_test_data['sentence'], sst_test_data['label'],
                                    cola_test_data[0], cola_test_data[1])
        test_generator = torch.utils.data.DataLoader(
            test_set, **params)

        if args.type == 'lstm':
            # TODO: implement both data and model
            model = ...
            pass
        elif args.type == 'bert':
            # TODO: implement both data and model
            # https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
            model = ...
            pass

        # Loop over epochs
        for epoch in range(max_epochs):
            # Training
            for local_batch, local_labels in training_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)

                # Model computations
                [...]

            # Validation
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in validation_generator:
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.to(
                        device), local_labels.to(device)

                    # Model computations
                    [...]
    return 0
