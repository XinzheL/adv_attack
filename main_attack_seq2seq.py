# import fairseq
import torch
import fairseq
# from adversarial import adversaries, adversarial_criterion
from fairseq.utils import resolve_max_positions
from utils.fairseq_models import load_task_and_model
from copy import deepcopy
import logging

log_file = 'attack_info.txt'
logging.basicConfig(filename=log_file, level=logging.DEBUG)
open(log_file, 'w').close()


# load task and model
restore_from_hub = True
# To-Do: enter this argument fom cmd
if restore_from_hub:
    input_args = [
        '--arch', 'transformer_vaswani_wmt_en_de_big',
        '--restore-file', 'checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt',
        '--bpe', 'subword_nmt',
        # '--bpe-codes', 'checkpoints/wmt14.en-fr.joined-dict.transformer/bpecodes',
        '--tokenizer', 'moses',
        '--source-lang', 'en',  '--target-lang', 'fr', # To-do: may be automatically inferred
        'checkpoints/wmt14.en-fr.joined-dict.transformer/',
        ] # last one is for argument name :`data`, but here use to load task dictionary

else:
    input_args =[
        '--arch', 'transformer_vaswani_wmt_en_de_big',
        '--restore-file', 'checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt',
        '--bpe', 'subword_nmt',
        # '--bpe-codes', 'checkpoints/wmt14.en-fr.joined-dict.transformer/bpecodes',
        '--tokenizer', 'moses',
        '--source-lang', 'en',  '--target-lang', 'fr', # To-do: may be automatically inferred
        'checkpoints/wmt14.en-fr.joined-dict.transformer/',
        ] # last one is for argument name :`data`, but here use to load task dictionary


# set parser with args related to translation task
parser = fairseq.options.get_training_parser(default_task='translation')
args = fairseq.options.parse_args_and_arch(parser, input_args=input_args)



if restore_from_hub:
    # Load a transformer trained on WMT'16 En-De
    en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer=args.tokenizer, bpe=args.bpe)
    cfg, task, model, bpe = en2fr.cfg, en2fr.task, en2fr.models[0], en2fr.bpe
else:
    args.reset_optimizer = True
    args.reset_meters = True
    args.reset_dataloader = True
    args.reset_lr_scheduler = True
    args.max_sentences_valid = 1  # We attack batch size 1 at the moment
    # add the following two arguments
    args.path = args.restore_file
    args.beam = 1 # beam size 1 for inference on the model, could use higher
    task, model, bpe = model_utils.load_task_and_model(args)


from utils.attack_seq2seq_utils import *

# Attack
if restore_from_hub:
    left_pad_source = task.cfg.left_pad_source
    left_pad_target = task.cfg.left_pad_target
else:
    left_pad_source = task.args.left_pad_source
    left_pad_target = task.args.left_pad_target

# setup and load valid data as benign examples
data_path = 'data/en-fr/'
src_dataset = fairseq.data.indexed_dataset.make_dataset(data_path+'test.en', impl='raw', fix_lua_indexing=True,dictionary=task.src_dict)
tgt_dataset = fairseq.data.indexed_dataset.make_dataset(data_path+'test.fr', impl='raw', fix_lua_indexing=True,dictionary=task.tgt_dict)
eval_dataset = fairseq.data.LanguagePairDataset(
        src_dataset, src_dataset.sizes, task.src_dict,
        tgt_dataset, tgt_dataset.sizes, task.tgt_dict,
        left_pad_source=left_pad_source, # True
        left_pad_target=left_pad_target, # False
        align_dataset=None,
        append_bos=True
    )


# 1. initialize triggers
num_trigger_tokens = 3
assert bpe.encode("the") == "the"
trigger_token_ids = [task.source_dictionary.indices[bpe.encode("the")]] * num_trigger_tokens

from torch.utils.data import DataLoader
from copy import deepcopy
universal_perturb_batch_size = 128
dataloader = DataLoader(eval_dataset, batch_size=universal_perturb_batch_size, shuffle=True, collate_fn=eval_dataset.collater)

loss_log = []
for samples in dataloader:

  # 2. prepend/concatnate triggers to the batch
  bsz, max_len = samples['net_input']['src_tokens'].shape
  trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids)) 
  trigger_sequence_tensor = trigger_sequence_tensor.repeat(bsz, 1).cpu() # return shape: (bsz, universal_seq_len)
  original_tokens = samples['net_input']['src_tokens'].clone()
  samples['net_input']['src_tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1) # return shape: (bsz, universal_seq_len+max_len)


  # ##### Attack
  # # 3. setup adversarial loss function
  args.adv_criterion='all_bad_words'
  adv_criterion = AllBadWordsCriterion(args)

  # # 3. get (1) E and (2) gradient w.r.t. trigger embeddings for current batch
  emb_matrix = model.encoder.embed_tokens.weight
  handle = add_hooks(model, emb_matrix.shape[0]) # add required-gradient hooks to encoder wordpiece embeddings

  averaged_grad, logging_output = get_average_grad(model, samples, trigger_token_ids, adv_criterion) # return shape : (universal_seq_len, embsize)
  averaged_grad = averaged_grad.unsqueeze(0) # return shape : (1, universal_seq_len, embsize)
  # 4. linear approximation using E and gradient to decrease L_adv 
  # shape : (B:1, T-1: 26, |V|:43771)
  # args.adversary = 'brute_force'
  # adversary = adversaries.build_adversary(args, model, task)
  # pass the gradients to a particular attack to generate token candidates for each token
  # `emb_matrix` would be required from `adversary.model`
  cand_trigger_token_ids = adversary_forward(averaged_grad, emb_matrix, num_gradient_candidates=1)  # return shape : (universal_seq_len, num_candidates)

  # 5.(To Do): Tries all of the candidates and returns the trigger sequence with highest loss. (Beam search)
  # trigger_token_ids = all_attack_utils.get_best_candidates(model,
  #                                                 [batch],
  #                                                 trigger_token_ids,
  #                                                 cand_trigger_token_ids)
  trigger_token_ids = cand_trigger_token_ids.squeeze()
  print(logging_output['loss'])
  loss_log.append(logging_output['loss'])

with open("log.txt", "w") as output:
  output.write(str(loss_log))
