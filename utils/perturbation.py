import numpy as np
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# assert data['input_ids'].shape[0] == random_noise.shape[0]
# data['attention_mask'] = torch.cat((torch.ones((data['attention_mask'].shape[0],1)),data['attention_mask']), 1)
# data['input_ids'] = torch.cat((data['input_ids'][:, 0:1], random_noise, data['input_ids'][:, 1:]), 1)

