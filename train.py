import argparse
from tqdm.auto import tqdm
import pickle
import os
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description="Training...")
parser.add_argument('--modification', action='store_true', default=True)
parser.add_argument('--modification-file', type=str, default=None)
parser.add_argument('--use-the',action='store_true', default=False)
parser.add_argument('--first',action='store_true', default=False)
parser.add_argument('--num-training-steps',type=int, default=None)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint = "bert-base-uncased"
datasets = load_dataset("data_scripts/glue.py", "sst2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# load perturbation
if args.modification and os.path.isfile( args.modification_file):
    with open(args.modification_file, 'rb') as fp:
        m = pickle.load(fp)
    best_positions = []
    replacment_for_best_positions = []
    for i,j in m.values():
        if args.first:
            best_positions.append(2)
        else:
            best_positions.append(i)
        if args.use_the:
            replacment_for_best_positions.append(624)
        else:
            replacment_for_best_positions.append(j)
    best_positions = torch.tensor(best_positions, dtype=torch.int64).unsqueeze(1)
    replacment_for_best_positions = torch.tensor(replacment_for_best_positions, dtype=torch.int64).unsqueeze(1)

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device)

# prepare dataloader for batching and in-batch padding
batch_size = 32
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(
    tokenized_datasets["train"], shuffle=False, batch_size=batch_size, collate_fn=data_collator
)
eval_loader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, batch_size=batch_size, collate_fn=data_collator
)
data_iter = iter(train_loader)

# eval
def eval(eval_loader, model):
    total = 0
    correct = 0
    model.eval()
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        total += batch['labels'].size(0)
        correct += (predictions == batch['labels']).sum().item()

   
    return correct / total




# train
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 1
if args.num_training_steps is None:
    num_training_steps = num_epochs * len(train_loader)
else:
    num_training_steps = args.num_training_steps

progress_bar = tqdm(range(num_training_steps))
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps
# )

model.train()
eval_metrix = []
optim_steps = 0
for epoch in range(num_epochs):
    if optim_steps >= num_training_steps:
        break
    train_idx = 0
    for batch in train_loader:
        if args.modification:
            # update adv example into batch
           
            batch['input_ids'].scatter_(dim=1, index=best_positions[train_idx:(train_idx+batch_size)], src=replacment_for_best_positions[train_idx:(train_idx+batch_size)])
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optim_steps += 1
        # lr_scheduler.step()
        optimizer.zero_grad()
        eval_acc = eval(deepcopy(eval_loader), model)
        print(eval_acc)
        eval_metrix.append(eval_acc)
        progress_bar.update(1)
        train_idx += batch_size
        if optim_steps >= num_training_steps:
            break
        

with open(f"train_iter{num_training_steps}.txt", 'wb') as fp:
    pickle.dump(eval_metrix, fp)


