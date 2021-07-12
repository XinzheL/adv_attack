from typing import List, Iterator, Dict, Tuple, Any
from overrides import overrides
import random
from copy import deepcopy

import numpy
from nltk import pos_tag
import torch

import torch.optim as optim
from torch.utils import data

from allennlp.data import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.interpret.attackers import Hotflip
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token
from allennlp.data import Batch
from allennlp.nn import util
from allennlp.interpret.attackers.attacker import Attacker
import spacy
from allennlp.data.token_indexers import (
    ELMoTokenCharactersIndexer,
    TokenCharactersIndexer,
    SingleIdTokenIndexer,
)
from allennlp.interpret.attackers import utils
from allennlp.modules.token_embedders import Embedding
from attack_utils.allennlp_model import filter_instances, infer_instances, AllennlpDataset, get_gradients, _make_embedder_input, calculate_trigger_diversity, get_embedding_matrix



class UniversalAttack():
    """
    # Parameters

    predictor : `Predictor`
        The model (inside a Predictor) that we're attacking.  We use this to get gradients and
        predictions.
    universal_perturb_batch_size: `int` optional (default=`128`)
        This defines how many examples used to calculate average gradients for updating trigger 
        tokens once.
    num_trigger_tokens: `int` optional (default=`3`)
        This defines the length of trigger tokens.
    vocab_namespace : `str`, optional (default=`'tokens'`)
        We use this to know three things: (1) which tokens we should ignore when producing flips
        (we don't consider non-alphanumeric tokens); (2) what the string value is of the token that
        we produced, so we can show something human-readable to the user; and (3) if we need to
        construct a fake embedding matrix, we use the tokens in the vocabulary as flip candidates.
    max_tokens : `int`, optional (default=`5000`)
        This is only used when we need to construct a fake embedding matrix.  That matrix can take
        a lot of memory when the vocab size is large.  This parameter puts a cap on the number of
        tokens to use, so the fake embedding matrix doesn't take as much memory.
    """
    def __init__(self, predictor: Predictor,
        vocab_namespace: str = "tokens", 
        distributed: bool = False) -> None:
        
        # predictor-relevant
        self.predictor = predictor
        self._model = self.predictor._model
        self.cuda_device = next(self._model.named_parameters())[1].get_device()
        self.vocab = self._model.vocab
        self.namespace = vocab_namespace
        
        # Force new tokens to be alphanumeric
        self.invalid_replacement_indices: List[int] = []
        for i in self.vocab._index_to_token[self.namespace]:
            if not self.vocab._index_to_token[self.namespace][i].isalnum():
                self.invalid_replacement_indices.append(i)

        # for distributed
        self.distributed = distributed
        if self.distributed:
            from torch.nn import DataParallel
            self._model = DataParallel(self._model)

        
        

        

    @classmethod
    def prepend_batch(cls, instances, trigger_tokens=None, vocab=None):
        """
        trigger_tokens List[str] ：
        """
        instances_with_triggers = deepcopy(instances)
        for instance in instances_with_triggers: 
            if str(instance.fields['tokens'].tokens[0]) == '[CLS]':
                instance.fields['tokens'].tokens = [instance.fields['tokens'].tokens[0]] + \
                    [Token(token_txt) for token_txt in trigger_tokens] + \
                    instance.fields['tokens'].tokens[1:]
            else:
                instance.fields['tokens'].tokens = [Token(token_txt) for token_txt in trigger_tokens] + instance.fields['tokens'].tokens
            instance.fields['tokens'].index(vocab)
        
        return instances_with_triggers


    def attack_instances(
        self,
        instances: List[Instance],
        test_instances: List[Instance],
        triggers_txt: str,
        label_filter:int ,
        target_label: int,
        batch_size: int = 128,
        blacklist=List[Any],
        updatelist = None,
        first_cls = False,
        num_epoch: int =1,
        patient: int = 10,
        ) : 

        embedding_matrix: torch.Tensor = get_embedding_matrix(self._model,  ) # self.predictor._dataset_reader._token_indexers, self.namespace

        # initialize trigger tokens and metrix
        trigger_tokens = []
        update_idx = []
            
        for i, token in enumerate(triggers_txt.split(' ')):
            if token[0] == '{' and token[-1]=='}':
                trigger_tokens.append(token[1:-1])
                update_idx.append(i)
        if first_cls:
            update_idx = [idx+1 for idx in update_idx]

        trigger_diversity = None
        pertubated_accuracy = None
        pertubated_loss = None

        # get target label id
        if str(target_label) in self.vocab._token_to_index['labels'].keys(): # vocab responsible for label <-> label_id
            target_label = self.vocab._token_to_index['labels'][str(target_label)]
        # add blacklist 
        if updatelist is None:
            for token in blacklist:
                if token in self.vocab._token_to_index[self.namespace].keys():
                    idx = self.vocab._token_to_index[self.namespace][token]
                    self.invalid_replacement_indices.append(idx)
        else:
            for token in self.vocab._token_to_index[self.namespace].keys():
                if token not in updatelist:
                    idx = self.vocab._token_to_index[self.namespace][token]
                    self.invalid_replacement_indices.append(idx)
        
        
        # filter instances
        instances = filter_instances(instances, label_filter=label_filter)
        test_instances = filter_instances(test_instances, label_filter=label_filter)

        

        # initialize log for output
        log_trigger_tokens = [None] * (num_epoch+1)
        metrics_lst = [None] * (num_epoch+1)
        loss_lst = [None] * (num_epoch+1)
        diversity_lst = [None] * (num_epoch+1)
        
        # log for no triggers and initialized triggers
        orig_accuracy, orig_loss = infer_instances(test_instances, 
                                            self._model, 
                                            cuda_device=self.cuda_device, 
                                            distributed=self.distributed, 
                                            return_just_loss=True)
        prepended_test_instances = self.prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab)
        pertubated_accuracy, pertubated_loss = infer_instances(prepended_test_instances, 
                                            self._model, 
                                            cuda_device=self.cuda_device, 
                                            distributed=self.distributed, 
                                            return_just_loss=True)
                                            
                                            
    
        log_trigger_tokens[0] = ['', '-'.join(trigger_tokens)]
        metrics_lst[0] = [orig_accuracy, pertubated_accuracy]
        loss_lst[0] = [orig_loss, pertubated_loss]
        diversity_lst[0] = [-1, calculate_trigger_diversity(trigger_tokens, embedding_matrix, self.namespace, self.vocab)] # no triggers here, -1 means nothing 
        
        # sample batches, update the triggers, and repeat
        idx_for_best = 0
        worst_accuracy = 1
        idx_so_far = 0
        dataset = AllennlpDataset(instances, self.vocab)
        for epoch in range(num_epoch):
            if idx_so_far - idx_for_best  >= patient:
                break

            # initialize log list for this epoch
            log_trigger_tokens[epoch+1] = []
            metrics_lst[epoch+1] = []
            loss_lst[epoch+1] = []
            diversity_lst[epoch+1] = []

            batch_sampler = data.BatchSampler(data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
            for indices in batch_sampler:
                batch_copy = [dataset[i] for i in indices]
                if idx_so_far - idx_for_best  >= patient:
                    break
                    
                # prepend triggers
                batch_prepended = self.prepend_batch(batch_copy, trigger_tokens=trigger_tokens, vocab=self.vocab)

                
                # TODO: multiple candidates + beam search
                # Tries all of the candidates and returns the trigger sequence with highest loss.
                # trigger_token_ids = get_best_candidates(model,
                #                                                 [batch],
                #                                                 trigger_token_ids,
                #                                             cand_trigger_token_ids)
                
                # update triggers
                # index into tensor
                batch_prepended = Batch(batch_prepended)
                batch_prepended.index_instances(self._model.vocab)
                dataset_tensor_dict = util.move_to_device(batch_prepended.as_tensor_dict(), self.cuda_device)
                trigger_ids = [self.vocab.get_token_index(token_txt, namespace=self.namespace) for token_txt in trigger_tokens]
                new_trigger_ids = self.update_tokens(self.get_average_grad(self._model, dataset_tensor_dict, target_label, update_idx), trigger_ids, embedding_matrix)
                trigger_tokens = []
                for new_trigger_id in new_trigger_ids:
                    token_txt = self.vocab.get_token_from_index(new_trigger_id, namespace=self.namespace)
                    trigger_tokens.append(token_txt)


                # evaluate new triggers on test
                prepended_test_instances = self.prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab)
                pertubated_accuracy, pertubated_loss = infer_instances(prepended_test_instances, 
                                            self._model, 
                                            cuda_device=self.cuda_device, 
                                            distributed=self.distributed, 
                                            return_just_loss=True)
                             
                # if accuracy is worse
                if pertubated_accuracy <= worst_accuracy: 
                    worst_accuracy = pertubated_accuracy
                    idx_for_best = idx_so_far
                idx_so_far += 1

                # record metrics for output
                log_trigger_tokens[epoch+1].append('-'.join( trigger_tokens))
                metrics_lst[epoch+1].append(pertubated_accuracy)
                loss_lst[epoch+1].append(pertubated_loss)
                new_trigger_diversity = calculate_trigger_diversity(trigger_tokens, embedding_matrix, self.namespace, self.vocab)
                diversity_lst[epoch+1].append(new_trigger_diversity)


        return log_trigger_tokens, diversity_lst, metrics_lst, loss_lst
    

    # TODO: extract following functions out of this class, i.e., no `self`
    def get_average_grad( self, model, dataset_tensor_dict, target_label, update_idx):
        # get gradient(L_adv(x, \Tilde(y)))
        grads, _ = get_gradients(model, dataset_tensor_dict, target_label) # [B, T, C]
        # average grad across batch size, result only makes sense for trigger tokens at the front
        averaged_grad = numpy.sum(grads, 0)[update_idx] # return shape : (num_trigger_tokens, embsize)

        return averaged_grad

    def update_tokens(self, grad, token_ids, embedding_matrix):

        new_token_ids = [None] * len(token_ids)
        # pass the gradients to a particular attack to generate substitute token for each token.
        for index_of_token_to_flip in range(len(token_ids)):
            
            # Get new token using taylor approximation.
            new_token_id = self._first_order_taylor(
                grad[index_of_token_to_flip, :], 
                torch.from_numpy(numpy.array(token_ids[index_of_token_to_flip])),
                embedding_matrix= embedding_matrix
            )

            new_token_ids[index_of_token_to_flip] = new_token_id 
        return new_token_ids



    # this is totally same as HotFlip. (I put this method here for myself convenient checking)
    def _first_order_taylor(self, grad: numpy.ndarray, token_idx: torch.Tensor, embedding_matrix:  torch.Tensor, cuda_device=0) -> int:
        """
        The below code is based on
        https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/
        research/adversarial/adversaries/brute_force_adversary.py

        Replaces the current token_idx with another token_idx to increase the loss. In particular, this
        function uses the grad, alongside the embedding_matrix to select the token that maximizes the
        first-order taylor approximation of the loss.
        we want to minimize (x_perturbed-x) * grad(L_adv)
        
        """
        grad = util.move_to_device(torch.from_numpy(grad), cuda_device)
        if token_idx.size() != ():
            # We've got an encoder that only has character ids as input.  We don't curently handle
            # this case, and it's not clear it's worth it to implement it.  We'll at least give a
            # nicer error than some pytorch dimension mismatch.
            raise NotImplementedError(
                "You are using a character-level indexer with no other indexers. This case is not "
                "currently supported for hotflip. If you would really like to see us support "
                "this, please open an issue on github."
            )
        # if token_idx >= embedding_matrix.size(0):
        #     # This happens when we've truncated our fake embedding matrix.  We need to do a dot
        #     # product with the word vector of the current token; if that token is out of
        #     # vocabulary for our truncated matrix, we need to run it through the embedding layer.
        #     inputs = self._make_embedder_input([self.vocab.get_token_from_index(token_idx.item())])
        #     word_embedding = self.embedding_layer(inputs)[0]
        # else:
        word_embedding = torch.nn.functional.embedding(
            util.move_to_device(torch.LongTensor([token_idx]), cuda_device),
            embedding_matrix,
        )
        word_embedding = word_embedding.detach().unsqueeze(0)
        grad = grad.unsqueeze(0).unsqueeze(0)
        # solves equation (3) here https://arxiv.org/abs/1903.06620
        new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix)) # each instance: grad shape (seq_len, emb_size) ;emb_matrix shape (vocab_size, emb_size)
        prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embedding)).unsqueeze(-1)
        dir_dot_grad = new_embed_dot_grad - prev_embed_dot_grad # minimize (x_perturbed-x) * grad(L_adv)
        dir_dot_grad = dir_dot_grad.detach().cpu().numpy()   
        neg_dir_dot_grad = - dir_dot_grad  # maximize -(x_perturbed-x) * grad(L_adv)
        # Do not replace with non-alphanumeric tokens
        neg_dir_dot_grad[:, :, self.invalid_replacement_indices] = -numpy.inf
        best_at_each_step = neg_dir_dot_grad.argmax(2)
        return best_at_each_step[0].data[0]
        
        
   
           
 








        

