import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers import AutoConfig, AutoModelForPreTraining
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# -------------------------- LSTM + Word Embedding --------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# word_vecs = load_glove_vectors()
# pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)


def get_emb_matrix(word_vecs, word_counts, emb_size=50):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32')  # adding a vector for padding
    # adding a vector for unknown words
    W[1] = np.random.uniform(-0.25, 0.25, emb_size)
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25, 0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1
    return W, np.array(vocab), vocab_to_idx


def load_glove_vectors(glove_file="./data/glove.6B/glove.6B.50d.txt"):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors


def load_word2vec():
    from gensim.models import Word2Vec

    model = Word2Vec(reviews, size=100, window=5, min_count=5, workers=4)
    # gensim model created

    import torch

    weights = torch.FloatTensor(model.wv.vectors)
    embedding = nn.Embedding.from_pretrained(weights)


class InferClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, matrix_embeddings):
        """initializes a 2 layer MLP for classification.
        There are no non-linearities in the original code, Katia instructed us 
        to use tanh instead"""

        super(InferClassifier, self).__init__()

        # dimensionalities
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = 512

        # embedding
        self.embeddings = nn.Embedding.from_pretrained(matrix_embeddings)
        self.embeddings.requires_grad = False

        # creates a MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),  # not present in the original code.
            nn.Linear(self.hidden_dim, self.n_classes))

    def forward(self, sentence):
        """forward pass of the classifier
        I am not sure it is necessary to make this explicit."""

        # get the embeddings for the inputs
        u = self.embeddings(sentence)

        # forward to the classifier
        return self.classifier(x)


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        # x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

# ------------------------------ BERT + Tagger ------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


distilbert = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    distilbert, num_labels=2)

roberta = "roberta-base"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta)
roberta_model = AutoModelForSequenceClassification.from_pretrained(
    roberta, num_labels=2)

albert = "albert-base-v2"
albert_tokenizer = AutoTokenizer.from_pretrained(albert)
albert_model = AutoModelForSequenceClassification.from_pretrained(
    albert, num_labels=2)

bert_base = "bert-base-uncased"
bert_base_tokenizer = AutoTokenizer.from_pretrained(bert_base)
bert_base_model = AutoModelForSequenceClassification.from_pretrained(
    bert_base, num_labels=2)


def distilbert_preprocess_function(examples):
    return distilbert_tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_dataset_train = train_dataset.map(distilbert_preprocess_function, batched=True)
# tokenized_dataset_test = test_dataset.map(distilbert_preprocess_function, batched=True)
# training_args = TrainingArguments("test_trainer", per_device_train_batch_size=16, per_device_eval_batch_size=16, evaluation_strategy="epoch", learning_rate=1e-5, num_train_epochs=3)
# trainer = Trainer(model=distilbert_model, args=training_args, train_dataset=tokenized_dataset_train, eval_dataset=tokenized_dataset_test, compute_metrics=compute_metrics)
# trainer.train()


def roberta_preprocess_function(examples):
    return roberta_tokenizer(examples["text"], padding="max_length", truncation=True)

# --------------------------- Prompt + Large LMs ----------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# Large language model prompting
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens = ['<sep>']  # original bart has no separater tokens
tokenizer.add_tokens(special_tokens)


class GenerativeModel(nn.Module):
    def __init__(self, model_name, cache_dir, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_config = AutoConfig.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.model = AutoModelForPreTraining.from_pretrained(
            model_name, cache_dir=cache_dir, config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs,
                             attention_mask=batch.enc_attn,
                             decoder_input_ids=batch.dec_idxs,
                             decoder_attention_mask=batch.dec_attn,
                             labels=batch.lbl_idxs,
                             return_dict=True)

        loss = outputs['loss']

        return loss

    def predict(self, batch, num_beams=4, max_length=50):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=batch.enc_idxs,
                                          attention_mask=batch.enc_attn,
                                          num_beams=num_beams,
                                          max_length=max_length)

        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(
                outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()

        return final_output
