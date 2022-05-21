import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers import AutoConfig, AutoModelForPreTraining
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import namedtuple


# Dataset for LSTM + Embedding / BERT
class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, sst_X, sst_y, cola_X, cola_y, labels, max_len=10000):
        assert len(sst_X) == len(sst_y), "SST data not one to one"
        assert len(cola_X) == len(cola_y), "CoLA data not one to one"

        self.max_len = max_len

        min_samples = min(len(sst_X), len(cola_X))
        sst_X = sst_X[:min(min_samples, max_len)]
        sst_y = sst_y[:min(min_samples, max_len)]
        cola_X = cola_X[:min(min_samples, max_len)]
        cola_y = cola_y[:min(min_samples, max_len)]

        self.sst_X = sst_X
        self.sst_y = sst_y
        self.cola_X = cola_X
        self.cola_y = cola_y

    def __len__(self):
        return len(self.sst_X)

    def __getitem__(self, index):
        # Get element consisting of sst_X, sst_y, cola_X, and cola_y
        return (self.sst_X[index], self.sst_y[index], self.cola_X[index], self.cola_y[index])


# Dataset for prompting (everything below)
gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn',
                    'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 'infos']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[
                      None] * len(gen_batch_fields))


class GenDataset(Dataset):
    def __init__(self, tokenizer, max_length, data, label, max_data_count, max_output_length=10000, no_bos=False):
        self.tokenizer = tokenizer
        self.max_length = self.max_output_length = max_length
        if max_output_length is not None:
            self.max_output_length = max_output_length
        # if you use bart, then this should be False; if you use t5, then this should be True
        self.no_bos = no_bos
        self.data = []
        self.load_data(data, label, max_data_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_prompt(self, type):
        pass

    def load_data(self, data, label, prompt_type, max_data_count):

        for d in data:
            if len(self.data) == max_data_count:
                break
            cur_input = d + ' <sep> ' + self.get_prompt(prompt_type)

            self.data.append({
                'input': cur_input,
                'target': str(label),
                'info': prompt_type
            })

    def collate_fn(self, batch):
        input_text = [x['input'] for x in batch]
        target_text = [x['target'] for x in batch]

        # encoder inputs
        inputs = self.tokenizer(
            input_text, return_tensors='pt', padding=True, max_length=self.max_length)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(
            target_text, return_tensors='pt', padding=True, max_length=self.max_output_length)
        dec_idxs = targets['input_ids']
        batch_size = dec_idxs.size(0)
        dec_idxs[:, 0] = self.tokenizer.eos_token_id
        dec_attn = targets['attention_mask']

        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros(
            (batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(
            lbl_attn == 0, -100)  # ignore padding

        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()

        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs,
            infos=[x['info'] for x in batch]
        )
