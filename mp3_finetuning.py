import torch
import random
import numpy as np

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Random seed set as {seed}")

torch.cuda.empty_cache()

MAIN_DIR = "metamia"

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json
import pandas as pd

df = pd.read_csv(f"{MAIN_DIR}/train_data.csv")
df.head()

train_texts, train_labels = df['document'].values, df['label'].values

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

tokenized_texts = []
from tqdm import tqdm

max_seq_length = 512  # Maximum sequence length for BERT

for text in tqdm(train_texts):
    tokenized_texts.append(tokenizer(text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt'))



# Tokenize the texts and convert them to tensors
from sklearn.metrics import accuracy_score, f1_score, classification_report

input_ids = torch.cat([t['input_ids'] for t in tokenized_texts], dim=0)
attention_mask = torch.cat([t['attention_mask'] for t in tokenized_texts], dim=0)
labels = torch.tensor(train_labels)

# Create a dataset and data loader
dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16
lr = 1e-5

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# len(val_dataset)

# val_dataset[0]

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()
            print(labels.cpu())
            total_samples += labels.size(0)


    print(classification_report(predictions.cpu().numpy(), labels.cpu().numpy()))
    return total_loss / len(dataloader), correct_predictions / total_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2%}")
    model.save_pretrained(f"{MAIN_DIR}/fine_tuned_bert_epoch_{epoch+1}_lr_{lr}")

# Save the fine-tuned model
model.save_pretrained(f"{MAIN_DIR}/fine_tuned_bert")

# np.count_nonzero(labels.cpu().numpy() == 0)

c = 0
for i in labels.cpu().numpy():
  if i == 0:
    c += 1

print(c)
len(labels)

"""## Inference"""

df_test = pd.read_csv(f"{MAIN_DIR}/test_data.csv")
df_test.head()

texts_test = df_test['document'].values

# To use the fine-tuned model for inference:
loaded_model = BertForSequenceClassification.from_pretrained(f"{MAIN_DIR}/fine_tuned_bert")
loaded_model.to(device)

loaded_model.eval()

all_preds = []

with torch.no_grad():
  for text in tqdm(texts_test):
      tokenized_sentence = tokenizer(text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
      input_ids = tokenized_sentence["input_ids"].to(device)
      attention_mask = tokenized_sentence["attention_mask"].to(device)

      outputs = loaded_model(input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      predictions = torch.argmax(logits, dim=1)
      all_preds.extend(predictions.cpu().numpy())

import csv
with open(f'{MAIN_DIR}/metamia_results.csv', mode='w') as csv_file: # for mp3.1, use filename 'mp3.1_results.csv'
    writer = csv.writer(csv_file)
    writer.writerow(['label'])
    for item in all_preds:
        writer.writerow([item])