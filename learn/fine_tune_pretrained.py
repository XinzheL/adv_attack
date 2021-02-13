from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)


# data
text_batch = ["I love Pixar", "I don't care Pixar"]
labels = torch.tensor([1, 0]).unsqueeze(0)

# tokenize & indexing
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# training
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss # F.cross_entropy(outputs.logits, labels)
loss.backward()
optimizer.step()
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=100)

