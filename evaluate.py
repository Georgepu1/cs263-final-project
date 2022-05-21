import torch
import numpy as np
from sklearn.metrics import mean_squared_error


def train_lstm_model(model, train_dl, val_dl, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()

            # TODO: incorporate multitask loss
            loss = F.cross_entropy(y_pred, y)

            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = eval_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
                sum_loss/total, val_loss, val_acc, val_rmse))


def eval_metrics(model, dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred,
                            y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


def exact_match(pred_text, gold_text, skip_first=False):
    N = 0
    acc = 0
    # all words are outputted correctly
    perfect_matches = 0
    # number of skipped
    mismatch = 0
    for pred, gold in zip(pred_text, gold_text):
        split_pred_text = pred.split()
        split_gold_text = gold.split()
        if len(split_pred_text) == len(split_gold_text):
            cur_words = len(split_pred_text)
            total_correct = 0
            start = 0
            if skip_first:
                start = 1
                cur_words -= 1

            for i in range(start, len(split_pred_text)):
                if split_pred_text[i] == split_gold_text[i]:
                    total_correct += 1
            if total_correct == cur_words:
                perfect_matches += 1
            acc += (total_correct / cur_words)
            N += 1
        # case for skipping first word
        elif skip_first and len(split_pred_text) == len(split_gold_text) - 1:
            cur_words = len(split_pred_text) - 1
            total_correct = 0
            for i in range(len(split_pred_text)):
                if split_pred_text[i] == split_gold_text[i+1]:
                    total_correct += 1
            if total_correct == cur_words:
                perfect_matches += 1
            acc += (total_correct / cur_words)
            N += 1
        else:
            mismatch += 1

    return acc, perfect_matches, N, mismatch
