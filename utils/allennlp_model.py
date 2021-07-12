
from typing import List, Iterator, Dict, Tuple, Any
from torch.utils.data import Dataset, Sampler
import torch
from torch import backends
import pandas as pd

from allennlp.data.instance import Instance
from allennlp.data import Batch
from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BertPooler, CnnEncoder, LstmSeq2VecEncoder
from transformers import BertTokenizer

from my_library.models import MyPooler
from my_library.models.sst_classifier import SstClassifier
from my_library.modules import SelfAttentionEncoder

def construct_model(MODEL_TYPE, vocab, word_embedding_dim = 300):
    
    embedder_info, encoder_type = MODEL_TYPE.split("__", 1)
    embedder_type, embedder_name = embedder_info.split("--")
    
    if embedder_type == "pretrained_transformer" or embedder_type == "pretrained_transformer_static":
        vocab.add_transformer_vocab(BertTokenizer.from_pretrained(embedder_name), 'tags')
        token_embedder = PretrainedTransformerEmbedder(model_name=embedder_name, train_parameters=False)
        word_embedding_dim = 768
    elif embedder_type == "embedding" or embedder_type == "embedding_static":
        token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=word_embedding_dim)

    if encoder_type == "bert_pooler":
        encoder = MyPooler(pretrained_model=embedder_name, dropout=0.1, requires_grad=True)
    elif encoder_type == 'lstm':
        encoder = LstmSeq2VecEncoder(word_embedding_dim, hidden_size=512, num_layers=2)
    elif encoder_type == 'cnn':
        encoder = CnnEncoder(word_embedding_dim, num_filters=6)
    elif encoder_type == 'attention':
        
        embedding_dim = 300 if (embedder_type == "embedding" or embedder_type == "embedding_static") else 768
        num_heads = 5 if(embedder_type == "embedding" or embedder_type == "embedding_static")  else 8
        encoder = SelfAttentionEncoder(embedding_dim, None, num_heads)
        
    else:
        print(f'Invalid MODEL_TYPE {MODEL_TYPE}')
        exit()

    model = SstClassifier(vocab, BasicTextFieldEmbedder(token_embedders={'tokens': token_embedder}), encoder)
    model.cuda()

    return model



def load_vocab_and_model(model_dir_path, vocab_path=None):
    
    _, model_name = model_dir_path.rsplit('/', 1)
    # load vocab
    if vocab_path is None:
        vocab_path = model_dir_path + '/vocabulary'
    vocab = Vocabulary.from_files(vocab_path)

    # construct model arch
    model = construct_model(model_name, vocab)

    # load trained model
    with open(f"{model_dir_path}/best.th", 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model



def infer_instances(instances, model, 
    cuda_device=0, distributed=False, 
    return_just_loss=True):
    """ output accuracy and model output (dictionary with logits and loss) of given data for given models
    """
    
    if distributed:
        # assert model is DataParallel
        model.module.get_metrics(reset=True)
    else:
        model.get_metrics(reset=True)
    
    batch_size = len(instances)
    with torch.no_grad():
        # index for tensor
        dataset = Batch(instances)
        dataset.index_instances(model.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)

        # forward pass
        if distributed:
            outputs = model(**model_input)
        
        else:
            outputs = model(**model_input)

    torch.cuda.empty_cache() # in case of OOM error


    if distributed:
        if return_just_loss:
            return model.module.get_metrics()['accuracy'], float(outputs['loss'].mean())
        else:
            return model.module.get_metrics()['accuracy'], outputs 
        
    else:
        if return_just_loss:
            return model.get_metrics()['accuracy'], float(outputs['loss'])
        else:
            return model.get_metrics()['accuracy'], outputs


                

def get_attack_result(path, select_index=None, only_triggers=False, return_loss=False):
    result_df = pd.read_csv(path)[['iteration', 'triggers', 'accuracy', 'loss', 'diversity']]
    triggers_lst = list(result_df['triggers']) # only evaluate on the last one
    accuracy_lst = list(result_df['accuracy'])
    loss_lst = list(result_df['loss'])
    diversity_lst = list(result_df['diversity'])
    if select_index is None:
        select_index = result_df['accuracy'].argmin()
    
    if select_index == 0:
        ttr = -1
    else:
        ttr = len(set(triggers_lst[select_index].split('-')))

    if only_triggers:
        return triggers_lst[select_index]
    if return_loss:
        return triggers_lst[select_index], accuracy_lst[select_index], loss_lst[select_index], diversity_lst[select_index], ttr
    else:
        return triggers_lst[select_index], accuracy_lst[select_index], diversity_lst[select_index], ttr



def filter_instances(instances, label_filter=None):
        targeted_instances = []
        if label_filter is None:
            return instances
        for instance in instances:
            if instance['label'].label == str(label_filter) or instance['label'].label == label_filter:
                targeted_instances.append(instance)
        return targeted_instances


class AllennlpDataset(Dataset):
    """
    An `AllennlpDataset` is created by calling `.read()` on a non-lazy `DatasetReader`.
    It's essentially just a thin wrapper around a list of instances.
    """

    def __init__(self, instances: List[Instance], vocab: Vocabulary = None):
        self.instances = instances
        self.vocab = vocab

    def __getitem__(self, idx) -> Instance:
        if self.vocab is not None:
            self.instances[idx].index_fields(self.vocab)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[Instance]:
        """
        Even though it's not necessary to implement this because Python can infer
        this method from `__len__` and `__getitem__`, this helps with type-checking
        since `AllennlpDataset` can be considered an `Iterable[Instance]`.
        """
        yield from self.instances

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab



def get_gradients(model, dataset_tensor_dict, target_label=None, grad_fn=torch.nn.CrossEntropyLoss()) -> Tuple[Any, Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters

        instances : `List[Instance]`

        # Returns

        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.

        # Notes

        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients: List[Tensor] = []

        def _register_embedding_gradient_hooks( embedding_gradients):

            def hook_layers(module, grad_in, grad_out):
                grads = grad_out[0]
                embedding_gradients.append(grads)

            hool_handles = []
            embedding_layer = util.find_embedding_layer(model)
            hool_handles.append(embedding_layer.register_backward_hook(hook_layers))
            return hool_handles

        hook_handles: List[RemovableHandle] = _register_embedding_gradient_hooks(embedding_gradients)

        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            

            if target_label is not None:
                # TODO: use Adv loss explicitely rather than use training loss
                dataset_tensor_dict['label'] = torch.empty(dataset_tensor_dict['label'].shape, 
                                                            dtype=dataset_tensor_dict['label'].dtype, 
                                                            device=dataset_tensor_dict['label'].device).fill_(target_label)
                outputs = model.forward(**dataset_tensor_dict)  # type: ignore
                
                # outputs['logits']
                loss = grad_fn(outputs['logits'], dataset_tensor_dict['label'])

                # Zero gradients.
                # NOTE: this is actually more efficient than calling `self._model.zero_grad()`
                # because it avoids a read op when the gradients are first updated below.
                for p in model.parameters():
                    p.grad = None
                loss.backward()

        for hook in hook_handles:
            hook.remove()

        # restore the original requires_grad values of the parameters
        for param_name, param in model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return embedding_gradients[0].detach().cpu().numpy(), outputs



def get_input_score(grads, embeddings):
    """

    # Parameters

        grads : `numpy.ndarray` shape (bsz, seq_len, embsize)
        embeddings : `numpy.ndarray` shape (bsz, seq_len, embsize)

    # Returns

    """
    emb_product_grad = grads * embeddings # shape: (seq_len, embsize)
    aggregate_emb_product_grad = numpy.sum(emb_product_grad, axis=1) # shape: (seq_len)
    norm = numpy.linalg.norm(aggregate_emb_product_grad, ord=1) # 
    normalized_scores = [math.fabs(e) / norm for e in aggregate_emb_product_grad]

    return normalized_scores


def calculate_trigger_diversity(trigger_tokens=None, embedding_matrix=None, namespace=None, vocab=None):
        if len(trigger_tokens) == 1:
            return 0
        trigger_token_ids = [vocab.get_token_index(trigger_tokens[i], namespace=namespace) for i in range(len(trigger_tokens))]
        word_embedding = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids), embedding_matrix.cpu())
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        diversity_score = 1
        for i in range(len(trigger_tokens)):
            for j in range(i+1, len(trigger_tokens)):
                diversity_score *= cos(word_embedding[i], word_embedding[j])
        
        diversity_score = 1 - diversity_score
        return diversity_score.item()


def get_embedding_matrix( model, indexers=None, namespace=None):
        """
        For HotFlip, we need a word embedding matrix to search over. The below is necessary for
        models such as ELMo, character-level models, or for models that use a projection layer
        after their word embeddings.

        We run all of the tokens from the vocabulary through the TextFieldEmbedder, and save the
        final output embedding. We then group all of those output embeddings into an "embedding
        matrix".
        """
        embedding_layer: torch.nn.Module = util.find_embedding_layer(model)
        if isinstance(embedding_layer, (Embedding, torch.nn.modules.sparse.Embedding)):
            # If we're using something that already has an only embedding matrix, we can just use
            # that and bypass this method.
            return embedding_layer.weight

#         # We take the top `self.max_tokens` as candidates for hotflip.  Because we have to
#         # construct a new vector for each of these, we can't always afford to use the whole vocab,
#         # for both runtime and memory considerations.
#         all_tokens = list(model.vocab._token_to_index[namespace])[: self.max_tokens]
#         max_index = model.vocab.get_token_index(all_tokens[-1], namespace)
#         self.invalid_replacement_indices = [
#             i for i in self.invalid_replacement_indices if i < max_index
#         ]
# 
#         inputs = _make_embedder_input(model, all_tokens, indexers, namespace)
# 
#         # pass all tokens through the fake matrix and create an embedding out of it.
#         embedding_matrix = embedding_layer(inputs).squeeze()
#         return embedding_matrix

def _make_embedder_input( model, all_tokens: List[str], indexers, namespace, cuda_device=0) -> Dict[str, torch.Tensor]:
        inputs = {}
        # A bit of a hack; this will only work with some dataset readers, but it'll do for now.
        for indexer_name, token_indexer in indexers.items():
            if isinstance(token_indexer, SingleIdTokenIndexer):
                all_indices = [
                    model.vocab._token_to_index[namespace][token] for token in all_tokens
                ]
                inputs[indexer_name] = {"tokens": torch.LongTensor(all_indices).unsqueeze(0)}
            elif isinstance(token_indexer, TokenCharactersIndexer):
                tokens = [Token(x) for x in all_tokens]
                max_token_length = max(len(x) for x in all_tokens)
                # sometime max_token_length is too short for cnn encoder
                max_token_length = max(max_token_length, token_indexer._min_padding_length)
                indexed_tokens = token_indexer.tokens_to_indices(tokens, model.vocab)
                padding_lengths = token_indexer.get_padding_lengths(indexed_tokens)
                padded_tokens = token_indexer.as_padded_tensor_dict(indexed_tokens, padding_lengths)
                inputs[indexer_name] = {
                    "token_characters": torch.LongTensor(
                        padded_tokens["token_characters"]
                    ).unsqueeze(0)
                }
            elif isinstance(token_indexer, ELMoTokenCharactersIndexer):
                elmo_tokens = []
                for token in all_tokens:
                    elmo_indexed_token = token_indexer.tokens_to_indices(
                        [Token(text=token)], model.vocab
                    )["elmo_tokens"]
                    elmo_tokens.append(elmo_indexed_token[0])
                inputs[indexer_name] = {"elmo_tokens": torch.LongTensor(elmo_tokens).unsqueeze(0)}
            else:
                raise RuntimeError("Unsupported token indexer:", token_indexer)

        return util.move_to_device(inputs, cuda_device)



class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        random.seed(22)
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            select_class = self.keys[self.currentkey]
            select_idx_for_this_class = self.indices[self.currentkey]
            yield self.dataset[select_class][select_idx_for_this_class]
            # currectkey back to 0 if each calss has been selected
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx):
        return dataset.__getitem__(idx)['label']._label_id

    def __len__(self):
        return self.balanced_max*len(self.keys)


def get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False, beam_size=1):
    """"
    Args:
    trigger_token_ids : the list of candidate trigger token ids with shape (num_of_trigger_words,  num_of_candidates)
    per word)
    
    Return:
    the best new candidate trigger found by performing beam search in a left to right fashion.
    """

    from operator import itemgetter
    import heapq

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


