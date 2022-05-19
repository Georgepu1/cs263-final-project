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


# Dataset for prompting (everything below)

gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn',
                    'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 'infos']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[
                      None] * len(gen_batch_fields))


class GenDataset(Dataset):
    def __init__(self, tokenizer, max_length, data, replace_words, max_data_count, max_output_length=None, unseen_types=[], no_bos=False):
        self.tokenizer = tokenizer
        self.max_length = self.max_output_length = max_length
        if max_output_length is not None:
            self.max_output_length = max_output_length
        # if you use bart, then this should be False; if you use t5, then this should be True
        self.no_bos = no_bos
        self.data = []
        self.load_data(data, replace_words, unseen_types, max_data_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, data, replace_words, unseen_types, max_data_count, cased=True, max_attempts=10):

        for d in data:
            if len(self.data) == max_data_count:
                break
            # Format the data with prompt, random replacement words "from => to" and expected output
            replace_words = None
            attempts = 0
            while replace_words is None and attempts < max_attempts:
                attempts += 1
                cur_replace_words = random.sample(replace_words, 2)
                cur_split_data = d.split()

                if cased:
                    if (cur_replace_words[0] in cur_split_data or cur_replace_words[1] in cur_split_data):
                        replace_words = cur_replace_words
                elif (cur_replace_words[0].capitalize() in cur_split_data or cur_replace_words[1].capitalize() in cur_split_data or
                      cur_replace_words[0] in cur_split_data or cur_replace_words[1] in cur_split_data or
                      cur_replace_words[0].lower() in cur_split_data or cur_replace_words[1].lower() in cur_split_data):
                    replace_words = cur_replace_words

            if attempts >= 10 or replace_words is None:
                continue
            cur_input = d + ' <sep> ' + \
                'Replace {} with {}: '.format(
                    replace_words[0], replace_words[1])
            # change output to modify input val to expected output val for all variants
            # expected_output = d.replace(" " + replace_words[0].capitalize() + " ", " " + replace_words[1].capitalize() + " ")
            expected_output = d.replace(
                " " + replace_words[0] + " ", " " + replace_words[1] + " ")
            # expected_output = d.replace(" " + replace_words[0].lower() + " ", " " + replace_words[1].lower() + " ")

            self.data.append({
                'input': cur_input,
                'target': expected_output,
                'info': replace_words
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
