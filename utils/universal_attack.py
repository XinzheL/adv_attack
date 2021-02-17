

from allennlp.interpret.attackers import Hotflip
import numpy
import torch
# for batch
from copy import deepcopy
from allennlp.nn.util import move_to_device

# for grad()
import torch.optim as optim
from typing import List
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token

# batch data
from .allennlp_data import AllennlpDataset
from torch.utils.data import DataLoader
from allennlp.data import Batch
# from allennlp.data.data_loaders import DataLoader

from allennlp.nn import util



class UniversalAttack(Hotflip):
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
    def __init__(self, predictor: Predictor=None, trigger_tokens =None, universal_perturb_batch_size: int = 128,
     num_trigger_tokens: int = 3,
     vocab_namespace: str = "tokens", max_tokens: int = 5000) -> None:
        super(UniversalAttack, self).__init__(predictor, vocab_namespace, max_tokens)
        self.batch_size = universal_perturb_batch_size
        # initialize triggers which are concatenated to the input
        self.num_trigger_tokens = num_trigger_tokens
        
        
        if trigger_tokens is None:
            self.trigger_tokens = []
            for _ in range(self.num_trigger_tokens):
                self.trigger_tokens.append(Token("the"))

    @classmethod
    def prepend_batch(cls, instances, trigger_tokens=None, vocab=None):
        
        for instance in instances: 
            if str(instance.fields['tokens'].tokens[0]) == '[CLS]':
                instance.fields['tokens'].tokens = [instance.fields['tokens'].tokens[0]] + \
                    trigger_tokens + \
                    instance.fields['tokens'].tokens[1:]
            else:
                instance.fields['tokens'].tokens = trigger_tokens + instance.fields['tokens'].tokens
            instance.fields['tokens'].index(vocab)
        

        # prepend triggers to the batch
        # trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
        # trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(instances), 1)
        # batch['tokens']['tokens']['tokens'] = torch.cat((trigger_sequence_tensor, batch['tokens']['tokens']['tokens'].clone()), 1)
        
        return instances

    @classmethod
    def filter_instances(cls, instances, label_filter, vocab=None):
        # find examples to a specific class, e.g. only positive 
        # or negative examples in binary classification,
        # Notice that here we needs `_label_id` corresponding to
        # index in the vocalbuary instead of raw label
        targeted_instances = []
        for instance in instances:
            instance.index_fields(vocab)
            if instance['label']._label_id == label_filter:
                targeted_instances.append(instance)
        return targeted_instances

    def attack_instances(
        self,
        instances: List[Instance],
        test_data: List[Instance],
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: List[str] = None,
        target: JsonDict = None,
        num_epoch=5,
        vocab_namespace='tokens',
        label_filter:int = 1
        ) : # -> JsonDict
        if self.embedding_matrix is None:
            self.initialize()

        # by default, universal attack would like to flip `label_filter` to other 
        # (in binary, it always could be considered as targeted attack.)
        # However, for multi-classification, we could specify which class it targets to flip
        sign = -1 if target is None else 1
        
        # label_filter 1 = "0" = neg
        # so neg -> pos
        targeted_instances = self.filter_instances(instances, label_filter=label_filter, vocab=self.vocab)
        targeted_test = self.filter_instances(test_data, label_filter=label_filter, vocab=self.vocab)


        # batches with size: universal_perturb_batch_size for the attacks.
        
        dataset = AllennlpDataset(targeted_instances, self.vocab)

        metrics_lst = [None] * (num_epoch+1)
        loss_lst = [None] * (num_epoch+1)
        log_trigger_tokens = []

        # record orginal metrics and loss
        accuracy, loss = self.evaluate_instances( targeted_test)
        metrics_lst[0] = [accuracy]
        loss_lst[0] = [loss]

        # sample batches, update the triggers, and repeat
        # for batch in iterator(targeted_dev_data, num_epochs=5, shuffle=True):
        for epoch in range(num_epoch):
            # print(f'Epoch:{epoch}')
            metrics_lst[epoch+1] = []
            loss_lst[epoch+1] = []
            
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda b: b)
            for batch in dataloader:
                
                batch_copy = deepcopy(batch)

                # set the labels equal to the target (backprop from the target class, not model prediction)
                #batch_copy[0]['label'] = int(target_label) * torch.ones_like(batch_copy[0]['label']).cuda()
                batch_prepended = self.prepend_batch(batch_copy, trigger_tokens=self.trigger_tokens, vocab=self.vocab)

                
                
                # we got gradients for all positions but only use the first `self.num_trigger_tokens` positions
                # Also, if needed, we could record the metrics like accuracy during the forward pass
                
                grads, _ = self.predictor.get_gradients(batch_prepended)
                grads = grads[grad_input_field] # [B, T, C]


                # average grad across batch size, result only makes sense for trigger tokens at the front
                averaged_grad = numpy.sum(grads, 0)
                averaged_grad = averaged_grad[0:self.num_trigger_tokens] # return shape : (num_trigger_tokens, embsize)

                new_trigger_tokens = [None] * self.num_trigger_tokens
                # pass the gradients to a particular attack to generate substitute token for each token.
                for index_of_token_to_flip in range(self.num_trigger_tokens):
                    original_id_of_token_to_flip = self.vocab.get_token_index(str(self.trigger_tokens[index_of_token_to_flip]), namespace=vocab_namespace)
                    # Get new token using taylor approximation.
                    trigger_indexed_token = self._first_order_taylor(
                        averaged_grad[index_of_token_to_flip, :], 
                        torch.from_numpy(numpy.array(original_id_of_token_to_flip)), sign
                    )
                    token_txt = self.vocab.get_token_from_index(trigger_indexed_token, namespace=vocab_namespace)
                    new_trigger_tokens[index_of_token_to_flip] = Token(token_txt)
                log_trigger_tokens.append(self.trigger_tokens)
                self.trigger_tokens = new_trigger_tokens
                    
                
                # TODO: multiple candidates + beam search
                # Tries all of the candidates and returns the trigger sequence with highest loss.
                # trigger_token_ids = get_best_candidates(model,
                #                                                 [batch],
                #                                                 trigger_token_ids,
                #                                             cand_trigger_token_ids)
        
                accuracy, loss = self.evaluate_instances( self.prepend_batch(targeted_test, trigger_tokens=self.trigger_tokens, vocab=self.vocab))
                metrics_lst[epoch+1].append(accuracy)
                loss_lst[epoch+1].append(loss)
        log_trigger_tokens.append(self.trigger_tokens)
        return loss_lst, metrics_lst, log_trigger_tokens

    def evaluate_instances(self, targeted_instances):
        self.predictor._model.get_metrics(reset=True)
        batch_size = len(targeted_instances)
        with torch.no_grad():
            dataset = Batch(targeted_instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
            outputs = self.predictor._model(**model_input)

        return self.predictor._model.get_metrics()['accuracy'], float(outputs['loss'])

    @classmethod
    def evaluate_instances_cls(cls, targeted_instances, model, vocab, cuda_device=0):
        model.get_metrics(reset=True)
        batch_size = len(targeted_instances)
        with torch.no_grad():
            dataset = Batch(targeted_instances)
            dataset.index_instances(vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = model(**model_input)

        return model.get_metrics()['accuracy'], float(outputs['loss'])


    def _first_order_taylor(self, grad: numpy.ndarray, token_idx: torch.Tensor, sign: int) -> int:
        """
        The below code is based on
        https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/
        research/adversarial/adversaries/brute_force_adversary.py

        Replaces the current token_idx with another token_idx to increase the loss. In particular, this
        function uses the grad, alongside the embedding_matrix to select the token that maximizes the
        first-order taylor approximation of the loss.
        """
        grad = util.move_to_device(torch.from_numpy(grad), self.cuda_device)
        if token_idx.size() != ():
            # We've got an encoder that only has character ids as input.  We don't curently handle
            # this case, and it's not clear it's worth it to implement it.  We'll at least give a
            # nicer error than some pytorch dimension mismatch.
            raise NotImplementedError(
                "You are using a character-level indexer with no other indexers. This case is not "
                "currently supported for hotflip. If you would really like to see us support "
                "this, please open an issue on github."
            )
        if token_idx >= self.embedding_matrix.size(0):
            # This happens when we've truncated our fake embedding matrix.  We need to do a dot
            # product with the word vector of the current token; if that token is out of
            # vocabulary for our truncated matrix, we need to run it through the embedding layer.
            inputs = self._make_embedder_input([self.vocab.get_token_from_index(token_idx.item())])
            word_embedding = self.embedding_layer(inputs)[0]
        else:
            word_embedding = torch.nn.functional.embedding(
                util.move_to_device(torch.LongTensor([token_idx]), self.cuda_device),
                self.embedding_matrix,
            )
        word_embedding = word_embedding.detach().unsqueeze(0)
        grad = grad.unsqueeze(0).unsqueeze(0)
        # solves equation (3) here https://arxiv.org/abs/1903.06620
        new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, self.embedding_matrix))
        prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embedding)).unsqueeze(-1)
        neg_dir_dot_grad = sign * (prev_embed_dot_grad - new_embed_dot_grad)
        neg_dir_dot_grad = neg_dir_dot_grad.detach().cpu().numpy()
        # Do not replace with non-alphanumeric tokens
        neg_dir_dot_grad[:, :, self.invalid_replacement_indices] = -numpy.inf
        best_at_each_step = neg_dir_dot_grad.argmax(2)
        return best_at_each_step[0].data[0]


#         torch.topk(torch.Tensor(neg_dir_dot_grad), 3, dim=2)[1][0,0,:].tolist()
# list(numpy.argsort(neg_dir_dot_grad, axis=2)[0][0][::-1][:3])
#            
 








        

