import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import trange

import re

import time
import torch.nn as nn

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def stratify(data, length):
    positive = np.where(data == 1)[0]
    negative = np.where(data == 0)[0]

    draw_p = positive[np.random.permutation(len(positive))[:length]]
    draw_n = negative[np.random.permutation(len(negative))[:length]]

    draw = np.hstack([draw_p, draw_n])
    np.random.shuffle(draw)

    return draw[:int(length * 1.5)]


data = pd.read_csv('Reviews.csv')
data = data.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"})
data = data[data['HelpfulnessNumerator'] <= data['HelpfulnessDenominator']]
data = data.dropna()

X, y = data['Text'][:100000], data['Score'][:100000]
y[y<4] = 0 # negative class
y[y>=4] = 1 # positive class
y = np.array(y)
print('total:', len(y))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = []

start = time.time()
for i, sentence in enumerate(X):
    if i % 10000 == 0:
        print(i, 'Time taken:', time.time() - start)
        start = time.time()
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)  # Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)  # Removing Punctuations
    tokenized_texts.append(tokenizer.tokenize(' '.join(['[CLS]', sentence, '[SEP]'])))

input_ids = []
attention_masks = []
labels = []

count = 0
for i in range(len(tokenized_texts)):
    if len(tokenized_texts[i]) < 128:
        input_ids.append(np.pad(np.array(tokenizer.convert_tokens_to_ids(tokenized_texts[i])), (0, 128 - len(tokenized_texts[i]))))
        attention_masks.append([float(j>0) for j in input_ids[count]])
        labels.append(y[count])
        count += 1

#splits data

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels,
                                                            random_state=42, test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=42, test_size=0.2)

train_masks, val_masks, _, _ = train_test_split(train_masks, train_inputs,
                                             random_state=42, test_size=0.2)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels,
                                                            random_state=42, test_size=0.2)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

draw_train = stratify(train_labels, 5000)
draw_val = stratify(val_labels, 500)
draw_test = stratify(test_labels, 500)

train_inputs = [train_inputs[i] for i in draw_train]
train_masks = [train_masks[i] for i in draw_train]
train_labels = [train_labels[i] for i in draw_train]

val_inputs = [val_inputs[i] for i in draw_val]
val_masks = [val_masks[i] for i in draw_val]
val_labels = [val_labels[i] for i in draw_val]

test_inputs = [test_inputs[i] for i in draw_test]
test_masks = [test_masks[i] for i in draw_test]
test_labels = [test_labels[i] for i in draw_test]

p = np.sum(train_labels) / len(train_labels)
print(np.sum(train_labels) / len(train_labels))
print(np.sum(val_labels) / len(val_labels))
print(np.sum(test_labels) / len(test_labels))

print(train_labels[:10])
print(val_labels[:10])


train_inputs = torch.tensor(train_inputs)
val_inputs = torch.tensor(val_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)
test_masks = torch.tensor(test_masks)

# Select a batch size for training.

# Create an iterator of our data with torch DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16, sampler=RandomSampler(train_data))
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=16, sampler=SequentialSampler(val_data))
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16, sampler=SequentialSampler(test_data))

# model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

print('model 1:', len(train_dataloader))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(model.parameters(),
                     lr=2e-6,
                     warmup=0.1)

train_loss_set = []
# Number of training epochs
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# BERT training loop
start = time.time()
for _ in trange(epochs, desc="Epoch"):

    ## TRAINING

    model.train()
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        if step % 100 == 0:
            print(step, 'time:', time.time() - start)
            start = time.time()
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    ## VALIDATION

    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in val_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += len(b_labels)
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

'''
## Prediction on test set
# Put model in evaluation mode
model.eval()
# Tracking variables
predictions, true_labels = [], []
# Predict
for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

print('Classification accuracy using BERT Fine Tuning: ', 1.0 * flat_true_labels / (1.0 * flat_predictions))
'''