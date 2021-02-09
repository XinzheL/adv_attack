# model arch/metric setup
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, BertPooler
from torch.nn import LSTM, Linear, CrossEntropyLoss
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary


# for training 
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
import torch.optim as optim
from allennlp.training.trainer import Trainer

from torch import save, load
import torch
from typing import Dict, Union, Optional

class SSTClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = CrossEntropyLoss()

    def forward(self, tokens: Dict[str, Dict[str, torch.Tensor]], \
        label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens) # shape: (B, T)
        embeddings = self.word_embeddings(tokens) # shape: (B, T, C)
        encoder_out = self.encoder(embeddings, mask) # shape: (B, hidden)
        logits = self.linear(encoder_out) # shape: (B, 2)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

class BertForClassification(Model):
    def __init__(self, vocab: Vocabulary, pretrained_embeddings, encoder):
        super().__init__(vocab)
        self.pretrained_embeddings = pretrained_embeddings
        self.encoder = encoder
        self.linear = Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = CrossEntropyLoss()

    def forward(self, \
        tokens:Dict[str, Dict[str, torch.Tensor]], \
        label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # indexer logic: e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" .
        token_ids = tokens['tokens']['tokens'] # shape: (B, T, C)
        mask = get_text_field_mask(tokens) # shape: (B, T)
        embeddings = self.pretrained_embeddings.forward(token_ids=token_ids, mask=mask) # shape: (B, T, C)
        encoder_out = self.encoder(embeddings, mask=mask) # shape: (B, C) for [CLS]
       
        logits = self.linear(encoder_out) # shape: (B, 2)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}


def train_sst_model(output_dir, READER_TYPE='pretrained', \
    Model_TYPE='pretrained',
    EMBEDDING_TYPE = None,  \
    pretrained_model = 'bert-base-uncased'):
    from .allennlp_data import load_sst_data
    from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
    
    # define output path
    model_path = output_dir + "model.th"
    vocab_path = output_dir +  "vocab"

    # 1. read data
    reader, train_data = load_sst_data('train', READER_TYPE=READER_TYPE, pretrained_model = pretrained_model)
    _, dev_data = load_sst_data('dev', READER_TYPE=READER_TYPE, pretrained_model = pretrained_model)
    
    if Model_TYPE == "pretrained":

        # 2. token embedding
        #vocab = Vocabulary.empty()
        vocab = Vocabulary.from_instances(dev_data) # TODO: just construct namespace for labels fields 
        reader._token_indexers['tokens']._add_encoding_to_vocabulary_if_needed(vocab)
        token_embedding = PretrainedTransformerEmbedder(model_name=pretrained_model)
        # 3. seq2vec encoder
        encoder = BertPooler(pretrained_model=pretrained_model)

        # 4. construct model
        model = BertForClassification(vocab, token_embedding, encoder)
        model.cuda()
    else:
        # 2. token embedding
        vocab = Vocabulary.from_instances(train_data)
        vocab_size = vocab.get_vocab_size('tokens')
        word_embedding_dim = 300
        if EMBEDDING_TYPE is None:
            token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)
        elif EMBEDDING_TYPE == "w2v":
            
            embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
            weight = _read_pretrained_embeddings_file(embedding_path,
                                                    embedding_dim=word_embedding_dim,
                                                    vocab=vocab,
                                                    namespace="tokens")
            token_embedding = Embedding(num_embeddings=vocab_size,
                                        embedding_dim=word_embedding_dim,
                                        weight=weight,
                                        trainable=False)
            from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
            word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

            
        
        # 3. seq2vec encoder
        if Model_TYPE == 'lstm':
            encoder = PytorchSeq2VecWrapper(LSTM(word_embedding_dim,
                                                        hidden_size=512,
                                                        num_layers=2,
                                                        batch_first=True))
            # 4. construct model
            model = SSTClassifier(word_embeddings, encoder, vocab)
            model.cuda()

            
    # 5. train model from scratch and save its weights
    
    
    from .allennlp_data import AllennlpDataset
    train_data = AllennlpDataset(train_data, vocab)
    dev_data = AllennlpDataset(dev_data, vocab)
    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    
    from allennlp.data.data_loaders import DataLoader
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    trainer = build_trainer(model, model_path, train_loader, dev_loader)

    with open(model_path, 'wb') as f:
        save(model.state_dict(), f)
    vocab.save_to_files(vocab_path)
    
import allennlp
from typing import Tuple
from allennlp.data.data_loaders import DataLoader
# The other `build_*` methods are things we've seen before, so they are
# in the setup section above.
def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=8, shuffle=False)
    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    from allennlp.training.trainer import GradientDescentTrainer
    from allennlp.training.optimizers import AdamOptimizer
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


def load_sst_model(file_dir, EMBEDDING_TYPE = 'w2v'):
    # load vocab and model
    from allennlp.data.vocabulary import Vocabulary
    
    vocab_path = file_dir + EMBEDDING_TYPE + "_" + "vocab"
    vocab = Vocabulary.from_files(vocab_path)
    vocab_size = vocab.get_vocab_size('tokens')

    model_path = file_dir + EMBEDDING_TYPE + "_" + "model.th"
    word_embedding_dim = 300
    word_embeddings = BasicTextFieldEmbedder({"tokens": Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)})
    encoder = PytorchSeq2VecWrapper(LSTM(word_embedding_dim,
                                                    hidden_size=512,
                                                    num_layers=2,
                                                    batch_first=True))
    model = SSTClassifier(word_embeddings, encoder, vocab)
    with open(model_path, 'rb') as f:
        model.load_state_dict(load(f))
    model.train().cuda() # rnn cannot do backwards in train mode

    return vocab, model

def get_embedding_matrix():
    named_modules = dict(model.named_modules())
    embedding_weight = named_modules['word_embeddings']._token_embedders['tokens'].weight
    return embedding_weight


def get_grad_for_emb(model, batch):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.

    Args:
    batch : see output of `allennlp_data.collate_fn`
    """
    from allennlp.modules.text_field_embedders import TextFieldEmbedder
    
    extracted_grads = []

    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    def add_hooks(model):
        """
        Finds the token embedding matrix on the model and registers a hook onto it.
        When loss.backward() is called, extracted_grads list will be filled with
        the gradients w.r.t. the token embeddings
        """
        for module in model.modules():
            if isinstance(module, TextFieldEmbedder):
                for embed in module._token_embedders.keys():
                    module._token_embedders[embed].weight.requires_grad = True
                hook = module.register_backward_hook(extract_grad_hook)
                return hook


    hook = add_hooks(model)
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # evaluate batch to get grad
    batch = move_to_device(batch, device=0)
    output_dict = model(batch['tokens'], batch['label'])
    loss = output_dict['loss']
    loss.backward()
    hook.remove()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    return extracted_grads[0].cpu()

    




from operator import itemgetter

import heapq
import numpy
import torch

def get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False, beam_size=1):
    """"
    Args:
    trigger_token_ids : the list of candidate trigger token ids with shape (num_of_trigger_words,  num_of_candidates)
    per word)
    
    Return:
    the best new candidate trigger found by performing beam search in a left to right fashion.
    """

    def evaluate_batch(model, batch, trigger_token_ids=None, snli=False):
        """
        Takes a batch of classification examples (SNLI or SST), and runs them through the model.
        If trigger_token_ids is not None, then it will append the tokens to the input.
        This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
        """
        batch = move_to_device(batch[0], device=0)
        if trigger_token_ids is None:
            model(batch['tokens'], batch['label'])
            return None
        else:
            trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
            trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['label']), 1).cuda()
            
            original_tokens = batch['tokens']['tokens']['tokens'].clone()
            batch['tokens']['tokens']['tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1)
            output_dict = model(batch['tokens'], batch['label']) 
            batch['tokens']['tokens']['tokens'] = original_tokens
            return output_dict


    def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False):
        """
        For a particular index, the function tries all of the candidate tokens for that index.
        The function returns a list containing the candidate triggers it tried, along with their loss.
        """
        if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
            print("Only 1 candidate for index detected, not searching")
            return trigger_token_ids
        model.get_metrics(reset=True)
        loss_per_candidate = []
        # loss for the trigger without trying the candidates
        curr_loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
        for cand_id in range(len(cand_trigger_token_ids[0])): # loop over num_candidates
            trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
            if cand_trigger_token_ids[index][cand_id] == 0:
                pass
            else:
                trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
            loss = evaluate_batch(model, batch, trigger_token_ids_one_replaced, snli)['loss'].cpu().detach().numpy()
            loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
        return loss_per_candidate

    # first round, no beams, just get the loss for each of the candidates in index 0.
    # return list with length num_candidates+1(1 refers to the original)
    loss_per_candidate = get_loss_per_candidate(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids, snli)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, cand,
                                                             cand_trigger_token_ids, snli))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]




