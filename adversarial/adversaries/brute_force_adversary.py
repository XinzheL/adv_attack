#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from . import BaseAdversary, register_adversary
from ..adversarial_utils import pairwise_distance


@register_adversary("brute_force")
class BruteForceAdversary(BaseAdversary):
    """This adversary just flips a fix amount of words so as to maximize the
    criterion"""

    def __init__(self, args, model, task):
        super().__init__(args, model, task)
        self.encoder = model.encoder
        self.max_swaps = args.max_swaps

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--normalize-directions",
            action="store_true",
            default=False,
            help="Take the dot product between the gradient and the normalized"
            "swap directions. When this is off the adversary will favor swaps"
            "with word vectors that are far away (because the magnitude of the"
            "dot product will be higher)",
        )

    def forward(self, input_gradients, num_gradient_candidates=None):
        """ 

        min [adv_token[i] - src_token[i]].dot(input_gradient) 

        Args:
        input_gradients: grad_adv[x_t] acieved by backward pass from L_adv to input embeddings
        """

        # 1. calculate important scores of replacing candidates of each input tokens
        # Purpose : replacing word x_t with word w 
        # Intuitive understanding : move the input in direction (w - x_t)
        # How ：find w in Vocab so to minimize L_adv(w)
        #       with first-order assumption/approximation, this equals to 
        #           minimize grad_adv[x_t]^T * (w - x_t)
        #       due to w x_t is constant, this equals to
        #           minimize grad_adv[x_t]^T * w
        
        # get the embeddings for vocabulary ; shape : (|V|, embsize)
        embedding_matrix = self.encoder.embed_tokens.weight

        # Take grad[x_i]^T * w_j for each position i in the source sentences
        # and each potential replacement w_j. 
        # shape : (B,T-1, |V|) <- (B, T-1, embsize) * (|V|, embsize) 
        # new_embed_dot_grad = input_gradients.bmm(embedding_matrix.t())
        gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (input_gradients, embedding_matrix.detach().cpu()))

        # 3. get candidates according to `gradient_dot_embedding_matrix` 
        # Option 1: take k candidates for each position as in Wallace, 2020
        # I prefer Option 2 due to we have already assume first-order approximation
        # Shape : (B. T-1, k)
        if num_gradient_candidates is not None:
            _, best_k_ids = torch.topk(gradient_dot_embedding_matrix * -1 , k=num_gradient_candidates, dim=2) 
            return best_k_ids.detach().cpu().numpy()[0] # shape: (T-1, k)
        
        # Option 2: take the best one for each position and then just consider how many tokens
        # could be swapped in the maximum
        else:
            score_at_each_step, best_at_each_step = torch.max(gradient_dot_embedding_matrix * -1, dim=2)
            return best_at_each_step

         
        