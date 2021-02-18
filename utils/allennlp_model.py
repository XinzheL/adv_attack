# model arch/metric setup
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, BertPooler
from torch.nn import LSTM, Linear, CrossEntropyLoss
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary, Instance



# for training 
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training import TensorBoardCallback
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
import torch.optim as optim
from allennlp.training.trainer import Trainer
from transformers import AdamW
from allennlp.training.optimizers import AdamOptimizer
import allennlp
from typing import Tuple
from allennlp.data.data_loaders import SimpleDataLoader
from .allennlp_data import MySimpleDataLoader

from torch import save, load
import torch
from typing import Dict, Union, Optional, List
from copy import deepcopy
import os

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
        # Option 1: use `TextFieldEmbedder`
        embeddings = self.pretrained_embeddings(tokens)
        mask = tokens['tokens']['mask'] #get_text_field_mask(tokens) # shape: (B, T)

        # Option 2: Just use `TokenEmbedder`
        # token_ids = tokens['tokens']['token_ids'] # shape: (B, T)
        # embeddings = self.pretrained_embeddings.forward(token_ids=token_ids, mask=mask) # shape: (B, T, C)
        
        encoder_out = self.encoder(embeddings, mask=mask) # shape: (B, C) for [CLS]
       
        logits = self.linear(encoder_out) # shape: (B, 2)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}


def train_sst_model(output_dir, train_data, dev_data, \
    MODEL_TYPE='finetuned_bert', \
    EMBEDDING_TYPE = None,  \
    pretrained_model = 'bert-base-uncased',
    num_epochs=3,
    bsz = 32,
    TRAIN_TYPE=None):

    
    
    if MODEL_TYPE == "finetuned_bert":

        # 2. token embedding
        # Problem: in huggingface, tokenizer is reponsible for tokenization and indexing which,
        #   in AllenNLP is functions of Vocabulary, Indexer and Tokenizer 
        # Soultion 1 : As normal steps in AllenNLP
        # extract Vocabulary from huggingface tokenizer with newly `get_vocab()` API and may still
        # use `SingleIdIndexer` instead of newly-added `PretrainedTransformerIndexer` shown below
        # Solution 2: relying on huggingface procedure 
        # (in AllenNLP, it has already induced `PretrainedTransformerIndexer` to do the thing 
        #  where `tokens_to_indices` independent of Vocab)
        # REMIND: `TextField.index()` use `Indexer.tokens_to_indices`, so I COULD just  ignore 
        #   extract vocab for AllenNLP vocab tokens namespace? 

        # Both indexing and adding special tokens are using huggingface tokenizer 
        # but at difference times. See Q1 and Q2.

        # Question 1: When to index TextField and LabelField? 
        # Answer: Handled in `DataLoader.iter_instances()` <- `DataLoader.iter()` 
        #   called by `Trainer._train_epoch()`
        #   for `TextField.index(vocab)`, see `PretrainedTransformerIndexer._extract_token_and_type_ids()` 
        #   which indeed uses huggingface tokenizer
        # Here I construct vocab for two reasons:
        # 1. for trainer API
        # 2. for indexing label field in Instance

        # Question 2: When special tokens are added?
        # Answer: During construction of instance  in `StanfordSentimentTreeBankDataReader.read()`
        
        # here, just construct namespace for labels fields 
        # TODO: Deprecate the use of LabelField and directly use label tensor 
        #   for forward and backward in Model
        vocab = Vocabulary.empty()
        from allennlp.data.fields import LabelField
        tmp_instances = [Instance(fields={'labels': LabelField('0', skip_indexing=False)}), \
            Instance(fields={'labels': LabelField('1', skip_indexing=False)})]
        vocab = Vocabulary.from_instances(tmp_instances) 
        # not sure the original idea using tags namespace in vocab, I choose ignoring it 
        #reader._token_indexers['tokens']._add_encoding_to_vocabulary_if_needed(vocab) 
        token_embedding = PretrainedTransformerEmbedder(model_name=pretrained_model, train_parameters=True)

        # 3. seq2vec encoder
        encoder = BertPooler(pretrained_model=pretrained_model, dropout=0.1, requires_grad=True)

        # 4. construct model
        
        model = BertForClassification(vocab, BasicTextFieldEmbedder({"tokens": token_embedding}), encoder)
        model.cuda()
    else:
        # 2. token embedding
        vocab = Vocabulary.from_instances(train_data)
        vocab_size = vocab.get_vocab_size('tokens')
        vocab_path =  "vocab"
        vocab.save_to_files(output_dir + vocab_path)
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
        
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
         
        # 3. seq2vec encoder
        if MODEL_TYPE == 'lstm':
            encoder = PytorchSeq2VecWrapper(LSTM(word_embedding_dim,
                                                        hidden_size=512,
                                                        num_layers=2,
                                                        batch_first=True))
            # 4. construct model
            model = SSTClassifier(word_embeddings, encoder, vocab)
            model.cuda()
            

            
    # 5. train model from scratch and save its weights
    
    
    # from .allennlp_data import AllennlpDataset
    # train_data = AllennlpDataset(train_data, vocab)
    # dev_data = AllennlpDataset(dev_data, vocab)
    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    
    
    
    
    # define optim
    if MODEL_TYPE == "finetuned_bert":
        optimizer = AdamW(model.parameters(),
                        lr=5e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )
    else:
        
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters)
    # from allennlp.training.learning_rate_schedulers import LearningRateScheduler
    # scheduler = LearningRateScheduler()
    if TRAIN_TYPE is None or TRAIN_TYPE == "error_max":
        train_loader, dev_loader = build_data_loaders(list(train_data), list(dev_data), vocab, bsz=bsz)
        trainer = build_trainer(model, output_dir, train_loader, dev_loader, num_epochs=num_epochs, optimizer=optimizer)
    elif TRAIN_TYPE == "error_min":
        if MODEL_TYPE == 'finetuned_bert':
            vocab_namespace='tags'
        elif MODEL_TYPE == 'lstm' and EMBEDDING_TYPE == 'w2v':
            vocab_namespace='tokens'
        train_loader = MySimpleDataLoader(train_data, batch_size=bsz, shuffle=False, vocab=vocab, )
        dev_loader = SimpleDataLoader(dev_data, batch_size=bsz, shuffle=False, vocab=vocab)
        trainer = build_error_min_unlearnable_trainer(model, output_dir, train_loader,\
            vocab, vocab_namespace, \
            dev_loader, num_epochs=num_epochs, optimizer=optimizer, )
    
    trainer.train()

    # define output path
    # model_path = "model.th"
    # with open(model_path, 'wb') as f:
    #     save(model.state_dict(), f)
    
    

# The other `build_*` methods are things we've seen before, so they are
# in the setup section above.
def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
    vocab: Vocabulary,
    bsz: int = 64
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = SimpleDataLoader(train_data, batch_size=bsz, shuffle=True, vocab=vocab)
    dev_loader = SimpleDataLoader(dev_data, batch_size=bsz, shuffle=False, vocab=vocab)
    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: allennlp.data.DataLoader,
    dev_loader: allennlp.data.DataLoader,
    num_epochs: int = 3,
    optimizer=None
    
) -> Trainer:
    
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        learning_rate_scheduler=None,
        patience=3,
        callbacks=[TensorBoardCallback(serialization_dir)]
    )
    return trainer

def build_error_min_unlearnable_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: allennlp.data.DataLoader,
    vocab,
    vocab_namespace,
    dev_loader: allennlp.data.DataLoader,
    num_epochs: int = 3,
    optimizer=None,
    
    
) -> Trainer:
    
    from .unlearnable_trainer import ErrorMinUnlearnableTrainer

    trainer = ErrorMinUnlearnableTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        vocab=vocab,
        vocab_namespace=vocab_namespace,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        learning_rate_scheduler=None,
        patience=3,
        callbacks=[TensorBoardCallback(serialization_dir)]
    )
    return trainer




def load_sst_model(file_dir, \
    model_path="model.th", \
    vocab_path='vocab', \
    MODEL_TYPE='finetuned_bert', \
    pretrained_model='bert-base-uncased'):
    # load vocab and model
    from allennlp.data.vocabulary import Vocabulary
    
    
    vocab_path = file_dir + vocab_path
    vocab = Vocabulary.from_files(vocab_path)
    

    if MODEL_TYPE == "finetuned_bert":
        
        # embedding
        token_embedding = PretrainedTransformerEmbedder(model_name=pretrained_model, train_parameters=True)
        # encoder
        encoder = BertPooler(pretrained_model=pretrained_model, dropout=0.1, requires_grad=True)
        # construct model
        model = BertForClassification(vocab, BasicTextFieldEmbedder({"tokens": token_embedding}), encoder)
        model.cuda()
    else:
        word_embedding_dim = 300
        vocab_size = vocab.get_vocab_size('tokens')
        word_embeddings = BasicTextFieldEmbedder({"tokens": Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)})
        encoder = PytorchSeq2VecWrapper(LSTM(word_embedding_dim,
                                                        hidden_size=512,
                                                        num_layers=2,
                                                        batch_first=True))
        model = SSTClassifier(word_embeddings, encoder, vocab)

    if os.path.isfile(file_dir + "model.th"):
        model_path = file_dir + "model.th"
    elif os.path.isfile(file_dir + "best.th"):
        model_path = file_dir + "best.th"
    else:
        pass

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




