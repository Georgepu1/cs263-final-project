# torch architecture


for _ in range(epochs):
    for (x1, y1), (x2, y2), ... in zip(ds_loader1, ds_loader2, ...):
        # Note can also set the data to a decide (cuda)
        m.zero_grad()
        m_outputs = [m(x) for x in [x1, x2, ...]]
        loss = c(m_outputs[0], y1) + \
            c(m_outputs[0], y2) + ...  # multitask loss
        loss.backward()
        # intermediate variabels stores embedding of x and computes
        # m_output.grad w.r.t. this and calculate the MSE of the m_output.grad(emb_x)
        # norm and 1.0 and use relu on; before you do the square, pass it through a relu
        # so everything les than 1.0 wont be counted to the square.
        # MSE (m_output.grad(x), 1.0) calulate gradient of M output w.r.t. x's embedding space
        # To avoid overfitting, calculating regularization term can use a varied version of x
        # instead of the original (e.g. add gaussian noise around embeddings of x); can
        # also minimize discrepancy on the two for robustness of model
        # Lipschitz-regularized loss
        o.step()
        # print statements/early stopping


# !pip install torch<=1.2.0
# !pip install torchtext
# %matplotlib inline

import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
	os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
