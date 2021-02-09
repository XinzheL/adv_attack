

#### adversary_criterion
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss



def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor
    
# @register_criterion("all_bad_words")
class AllBadWordsCriterion(_Loss):
    """This is essentially the negation of CrossEntropyCriterion.
    Instead of optimizing P[w1 AND w2 AND...] we optimize
    P[(NOT w1) AND (NOT w2) AND...]
    Notice that this is *not* the same as reversing the nll objective (which
    would mean optimizing P[(NOT w1) OR (NOT w2) OR...])
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, model, sample, padding_idx=1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # 1. Generate log probabilities according to the model
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # B x T x |V| -> (B*T) x |V|
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # 2. Compute log-probability of not producing any valid word
        target = model.get_targets(sample, net_output).view(-1)
        lp_targets = lprobs.index_select(dim=1, index=target)
        # Negation in the log-semiring (is that stable?)
        lp_not_targets = torch.log(1 - torch.exp(lp_targets))
        # Masking
        mask_pads = target.eq(padding_idx)
        lp_not_targets = lp_not_targets.masked_fill(mask_pads, 0)
        # Sum everything (=AND)
        if reduce:
            loss = lp_not_targets.sum()
        else:
            loss = lp_not_targets.view(net_output.size(0), -1).sum(-1)

        # Negate because we're minimizing the criterion
        loss = -loss

        # Negate the loss in order to maximize it by selecting candidates with top scores(scores would be calculated using grad x E) 
        # loss = -loss

        # if self.args.sentence_avg:
        #     sample_size = sample["target"].size(0)
        # else:
        #     sample_size = sample["ntokens"]

        logging_output = {
            "loss": item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            # "sample_size": sample_size,
        }
        return loss, logging_output

    


#### attack_utils.py
from copy import deepcopy
import logging
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
import fairseq


extracted_grads = []
def add_hooks(model, bpe_vocab_size):
    """ add hooks to embedding module for both encoder and decoder
        <- (id(model._modules['decoder']._modules['embed_tokens'])
                == id(model._modules['encoder']._modules['embed_tokens']))
        
        here shared source-target vocalbuary just works for some specific 
        language pair with similar granularity ( I think?? )
            
        state_dict['decoder.embed_tokens.weight'] 
        == state_dict['encoder.embed_tokens.weight']
        
    args:
        model
        bpe_vocab_size ( int ) : this is for encoder's embedding layer. 
            It actually comes from len(src_dict)  if you see `transformer` 
            source code for `build_embedding()`. And due to shared embedding,
            the hook also works for decoder
    
    """
    
    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads.append(grad_out[0])
        # extracted_emb_inp_grad.append(grad_in)
        
    
    hook_registered = False
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                module.weight.requires_grad = True
                # record the gradients w.r.t embeddings
                handle = module.register_backward_hook(extract_grad_hook)
                hook_registered = True
    return handle
    
    if not hook_registered:
        exit("Embedding matrix not found")

def get_average_grad(model, batch, trigger_token_ids, adversarial_criterion):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.

    Args:
      batch : tensor with shape (bsz, universal_seq_len+max_len)
    """
    # create an dummy(not for updating any parameters) optimizer to zero grad for backprop 
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()

    
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss, logging_output = forward_adversarial( model, batch, adversarial_criterion)
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].cpu()

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[0:len(trigger_token_ids)] # return just trigger grads
    return averaged_grad, logging_output

def forward_adversarial(model, sample, adversarial_criterion):
    """
    Args:
        adversarial_criterion: The objective to minimize for adversarial
        examples (eg. the log-likelihood)

    Return:
        loss: adversarial loss for minimization

     """
    # Set model to training mode
    # model.train()
    # But disable dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

    loss = None
    sample_size = 0
    logging_output = {
        "ntokens": sample["ntokens"] if sample is not None else 0,
        "nsentences": sample["target"].size(0) if sample is not None else 0,
    }
    
    # calculate loss and sample size
    (loss, logging_output_) = adversarial_criterion(
        model, sample
    )
    logging_output.update(logging_output_)


    return loss, logging_output

def adversary_forward(input_gradients, embedding_matrix, num_gradient_candidates=None):
    """ 

    min [adv_token[i] - src_token[i]].dot(input_gradient) 
    However, instead of finding adv_token minimizing formula above, this function output
    multiple substitution candidates according to the score adv_token[i].dot(input_gradient)

    Args:
    input_gradients: grad_adv[x_t] acieved by backward pass from L_adv to input embeddings 
        with shape (B, T, embsize)
    """
    
    # 1. calculate scores of replacing candidates of each input tokens
    # Purpose : replacing word x_t with word w 
    # Intuitive understanding : move the input in direction v
    # For attack by replacing token in original input: v = x_replaced - x_orig
    #       find w in Vocab so to minimize L_adv(x_replaced)
    #       with first-order assumption/approximation, this equals to 
    #           minimize L_adv(x_orig) + grad_adv^T * v
    #       since just change one token, consider token/inner dimension, it is
    #           minimize grad_adv[x_t]^T * (w - x_t)
    #       due to w x_t is constant, this equals to
    #           minimize grad_adv[x_t]^T * w
    # for univeral attack:
    #       find U = {w1,w2,w3} so to minimize L_adv()
    #       minimize nll_loss(x_adv), i.e. minimize e_adv_i * grad_nll_loss
    
    # get the embeddings for vocabulary ; shape : (|V|, embsize)

    # Take grad[x_i]^T * w_j for each position i in the source sentences
    # and each potential replacement w_j. 
    # shape : (B,T-1, |V|) <- (B, T-1, embsize) * (|V|, embsize) 
    # new_embed_dot_grad = input_gradients.bmm(embedding_matrix.t())
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (input_gradients, embedding_matrix.detach().cpu()))

    # 3. get candidates according to `gradient_dot_embedding_matrix` 
    # Option 1: take k candidates for each position as in Wallace, 2020
    # I prefer Option 2 due to we have already assume first-order approximation
    # Shape : (B. T, k)
    if num_gradient_candidates is not None:
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix * -1 , k=num_gradient_candidates, dim=2) 
        return best_k_ids.detach().cpu().numpy()[0] # shape: (T, k)
    
    # Option 2: take the best one for each position and then just consider how many tokens
    # could be swapped in the maximum
    else:
        score_at_each_step, best_at_each_step = torch.max(gradient_dot_embedding_matrix * -1, dim=2)
        return best_at_each_step

