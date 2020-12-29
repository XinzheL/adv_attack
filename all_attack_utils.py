from copy import deepcopy
import logging
logging.basicConfig(filename='debug.txt', level=logging.DEBUG)
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
import fairseq




extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

# returns the wordpiece embedding weight matrix
def get_embedding_weight(model, bpe_vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                return module.weight.detach().cpu()
    exit("Embedding matrix not found")




def get_user_input(trainer, bpe):
    user_input = input('Enter the input sentence that you to attack: ')
    if user_input.strip() == '':
        print("You entered a blank token, try again")
        return None

    # as in LanguagePairDataset() collate([{'id': 605, 'source': tensor([9860,  363,    2]), 'target': tensor([7022, 6838,    2])}], pad_idx=1, eos_idx=2)
    return process_sample(user_input, 
                          bpe,
                          trainer.task.source_dictionary,
                          trainer.get_model().encoder.embed_tokens.weight.shape[0])

def process_sample(sample, bpe, dictionary, bpe_vocab_size):
    # tokenize/vectorize input and get lengths 
    # 'i am sad'
    sample_bpe = bpe.encode(sample)
    # tensor([[  322,   106, 19454,     2]]) ps. 2 is `eos_index`
    sample_tokenized_bpe = dictionary.encode_line(sample_bpe).long().unsqueeze(dim=0)

    # check if the user input a token with is an UNK
    for token in sample_tokenized_bpe[0]:
        if torch.eq(token, bpe_vocab_size) or torch.gt(token, bpe_vocab_size): # >= max vocab size
            print('You entered an UNK token for your model, please try again. This usually occurs when (1) you entered '
                ' unicode or other strange symbols, (2) your model uses a lowercased dataset but you entered uppercase, or '
                ' (3) your model is expecting apostrophies as &apos; and quotes as &quot;')
            return None
    
    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])]) # [4]

    # build samples which is input to the model
    # ?? use of `ntokens`, same as `src_lengths`
    return {'net_input': 
                {'src_tokens': sample_tokenized_bpe, 'src_lengths': length_user_input}, 
            'ntokens': len(sample_tokenized_bpe[0])}


# runs the samples through the model and fills extracted_grads with the gradient w.r.t. the embedding
def get_loss_and_input_grad(trainer, samples, target_mask=None, no_backwards=False, reduce_loss=True):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() to turn off dropout and stuff    
    trainer.zero_grad() # optimizer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, _, = trainer.criterion(trainer.get_model(), sample, reduce=reduce_loss)    
    if not no_backwards:
        trainer.optimizer.backward(loss)
    return sample['net_input']['src_lengths'], loss.detach().cpu()


# take samples (which is batch size 1) and repeat it batch_size times to do batched inference / loss calculation
# for all of the possible attack candidates
def build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe, changed_positions=None, untouchable_token_blacklist=None, adversarial_token_blacklist=None, num_trigger_tokens=None):
    # copy and repeat the samples instead batch size elements
    samples_repeated_by_batch = deepcopy(samples)
    samples_repeated_by_batch['ntokens'] *= batch_size
    samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1)
    samples_repeated_by_batch['nsentences'] = batch_size

    all_inference_samples = [] # stores a list of batches of candidates
    all_changed_positions = [] # stores all the changed_positions for each batch element

    current_batch_size = 0
    current_batch_changed_position = []
    current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
    for index in range(len(candidate_input_tokens)): # for all the positions in the input
        for token_id in candidate_input_tokens[index]: # for all the candidates
            # for malicious nonsense
            if changed_positions is not None:
                # if we have already changed this position, skip
                if changed_positions[index]: 
                    continue
            # for universal triggers            
            if num_trigger_tokens is not None: 
                # want to change the last tokens, not the first, for triggers
                index_to_use = index - num_trigger_tokens - 1 # -1 to skip <eos>
            else:
                index_to_use = index

            # for targeted flips
            # don't touch the word if its in the blacklist
            if untouchable_token_blacklist is not None and current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] in untouchable_token_blacklist:
                continue
            # don't insert any blacklisted tokens into the source side
            if adversarial_token_blacklist is not None and any([token_id == blacklisted_token for blacklisted_token in adversarial_token_blacklist]): 
                continue

            original_token = deepcopy(current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use]) # save the original token, might be used below if there is an error
            current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = torch.LongTensor([token_id]).squeeze(0) # change one token

            # there are cases where making a BPE swap would cause the BPE segmentation to change.
            # in other words, the input we are using would be invalid because we are using an old segmentation
            # for these cases, we just skip those candidates            
            string_input_tokens = bpe.decode(trainer.task.source_dictionary.string(current_inference_samples['net_input']['src_tokens'][current_batch_size], None))
            retokenized_string_input_tokens = trainer.task.source_dictionary.encode_line(bpe.encode(string_input_tokens)).long().unsqueeze(dim=0)
            if torch.cuda.is_available() and not trainer.args.cpu:
                retokenized_string_input_tokens = retokenized_string_input_tokens.cuda()
            if len(retokenized_string_input_tokens[0]) != len(current_inference_samples['net_input']['src_tokens'][current_batch_size]) or \
                not torch.all(torch.eq(retokenized_string_input_tokens[0],current_inference_samples['net_input']['src_tokens'][current_batch_size])):
                # undo the token we replaced and move to the next candidate
                current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = original_token
                continue
                                    
            current_batch_size += 1
            current_batch_changed_position.append(index_to_use) # save its changed position

            if current_batch_size == batch_size: # batch is full
                all_inference_samples.append(deepcopy(current_inference_samples))
                current_inference_samples = deepcopy(samples_repeated_by_batch)
                current_batch_size = 0
                all_changed_positions.append(current_batch_changed_position)
                current_batch_changed_position = []

    return all_inference_samples, all_changed_positions

def get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=500, target_mask=None, increase_loss=False):
    # clear grads, compute new grads, and get candidate tokens
    global extracted_grads; extracted_grads = [] # clear old extracted_grads
    src_lengths, _ = get_loss_and_input_grad(trainer, samples, target_mask=target_mask) # gradient is now filled
    
    # 1. get grad of L_adv w.r.t. encoder embedding 
    # for models with shared embeddings, position 1 in extracted_grads will be the encoder grads, 0 is decoder
    # ?? how shared embeddings work during backward pass
    if len(extracted_grads) > 1:
        gradient_position = 1
    else:
        gradient_position = 0
    assert len(extracted_grads) <= 2 and len(extracted_grads[gradient_position]) == 1 # make sure gradients are not accumulating
    # first [] gets decoder/encoder grads specified by `gradient_position`, 
    # then gets ride of batch (we have batch size 1) specified by `[0]`
    # then we index into before the padding (though there shouldn't be any padding because we do batch size 1).
    # the -1 is to ignore the pad symbol.
    input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1].cpu() 
    input_gradient = input_gradient.unsqueeze(0) # unsqueeze the batch dim

    # 2. dot product a) grad of L_adv w.r.t. encoder embedding and b) emb
    # shape (1, 3, 1024) * (32768, 1024) - > (1,3,32768)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (input_gradient, embedding_weight))

    # 3. 
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_gradient_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_gradient_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    else:
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
        return best_at_each_step[0].detach().cpu().numpy()

def setup(input_args=None):
    """
    args :
        input_args (list)
            --arch
            --interactive-attacks
            --restore-file for model
            data
        

    return:
        args ( argparse.Namespace ) : see all_args.json
        
        
        trainer ( fairseq.trainer.Trainer ) :
        generator ( fairseq.sequence_generator.SequenceGenerator ) :
        embedding_weight ( torch.Tensor ) :
        itr ( list ) : all `None` if `input_args` set `--interactive-attacks`
        bpe (fairseq.data.encoders.subword_nmt_bpe.SubwordNMTBPE) :
    """

    # set parser with args related to translation task
    parser = options.get_training_parser(default_task='translation')

    parser.add_argument("--interactive-attacks", action='store_true')

    # add user-input args and parse these args
    # add model specific arguments according to `--arch` value
    # add {'criterion', 'tokenizer', 'bpe', 'optimizer', 'lr_scheduler'} specific arguments
    # add task-specific args
    # add model-arch specific arguments according to `--arch` value
    args = options.parse_args_and_arch(parser, input_args=input_args)

    # make sure everything is reset before loading the model
    args.reset_optimizer = True
    args.reset_meters = True
    args.reset_dataloader = True
    args.reset_lr_scheduler = True
    args.max_sentences_valid = 1  # We attack batch size 1 at the moment

    # add the following two arguments
    args.path = args.restore_file
    args.beam = 1 # beam size 1 for inference on the model, could use higher

    logging.debug(args)

    torch.manual_seed(args.seed)

    #*****Data******
    # setup task, model, loss function, and trainer 
    # 1). build fairseq.tasks.translation.TranslationTask 
    #    with attributes:
    #    q. self.src_dict : e.g. load dict.en.txt from args.data
    #      (actually model dir also has such dicts, this data path is unnecssary for interative attack)
    #    b. self.tgt_dict : e.g. load dict.de.txt from args.data 
    #    c. self.args : 
    task = tasks.setup_task(args)
    #*****

    # bpe: fairseq.data.encoders.subword_nmt_bpe.SubwordNMTBPE object
    # with an functional attribute bpe: subword_nmt.apply_bpe.BPE from subword library
    bpe = fairseq.data.encoders.build_bpe(args)

    assert bpe is not None
    #*****


    #*****Model*****
    # setup model arch by API from task or directly `models.build_model(arch)`
    # & and load checkpoint 
    # -----a. original: load_model_ensemble
    # models, _= checkpoint_utils.load_model_ensemble(args.path.split(':'), arg_overrides={}, task=task)
    # -----b. to load_model_ensemble_and_task
    # ensemble, _, _ = load_model_ensemble_and_task(filenames, arg_overrides, task)
    # -----1/2 add the following two lines
    # assert len(models) == 1 # Make sure you didn't pass an ensemble of models in
    # model = models[0]
    # -----c. to my simplified version
    def load_model(state_path, src_dict, tgt_dict):
        from torch.serialization import default_restore_location

        # 1). loads a checkpoint to CPU
        # `default_restore_location()` finds the working device matching 'cpu' from
        #  `_package_registry` list according to priority
        state = torch.load(
            state_path, map_location=lambda s, l: default_restore_location(s, 'cpu'),
        )
        state = checkpoint_utils._upgrade_state_dict(state) # upgrading for backward compatibility
        
        # 2). build a model (no care for parameters)

        # ----- task.build_model() 
        # ----- fairseq.models.build_model(args, self) 
        # ----- ARCH_MODEL_REGISTRY[args.arch].build_model(args, task) ?? how ARCH_MODEL_REGISTRY works globally
        # here model arch is saved in the state checkpoint model.pt file not from user input_args
        # model = task.build_model(state['args'])
        

        # it points to the same model in MODEL_REGISTRY in the memory
        # id(fairseq.models.ARCH_MODEL_REGISTRY['transformer_vaswani_wmt_en_de_big'])
        # =id(fairseq.models.ARCH_MODEL_REGISTRY['transformer_wmt_en_de'])
        # =id(fairseq.models.MODEL_REGISTRY['transformer'])
        model_arch = fairseq.models.ARCH_MODEL_REGISTRY[state['args'].arch] 
        model = model_arch.build_model(state['args'], src_dict, tgt_dict)
        
        # 3). load checkpoint parameters into `model`
        # as before, in pytorch there is no need for `args`, 
        # `args` is just for `prune_state_dict()`
        model.load_state_dict(state['model'], strict=True, args=args)

        return model, state['args']

    model, state_args = load_model(args.path, task.source_dictionary, task.target_dictionary)
    # logging.debug(model)
    # logging.debug(state_args)
    # load model to gpu
    if torch.cuda.is_available() and not args.cpu:
        assert torch.cuda.device_count() == 1 # only works on 1 GPU for now
        torch.cuda.set_device(0)
        model.cuda()

    model.make_generation_fast_(beamable_mm_beam_size=args.beam, need_attn=False)
    #*****

    #*****Loss Funcion******
    # setup criterion/loss function:   
    # ----- task.build_criterion(args)
    # ----- fairseq.criterions.build_criterion(args, self) 
    # ----- fairseq.criterions.cross_entropy.CrossEntropyCriterion(args, task)
    # only use args.sentence_avg, no other args or task
    criterion = fairseq.criterions.build_criterion(args, task) 
    #******

    # add hooks for embeddings, only add a hook to encoder wordpiece embeddings (not position)
    encoder_embed_layer = model.encoder.embed_tokens
    bpe_vocab_size = encoder_embed_layer.weight.shape[0]
    
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
        hook_registered = False
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == bpe_vocab_size:
                    module.weight.requires_grad = True
                    # record the gradients w.r.t embeddings
                    module.register_backward_hook(extract_grad_hook)
                    hook_registered = True
        if not hook_registered:
            exit("Embedding matrix not found")
    add_hooks(model, bpe_vocab_size) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model, bpe_vocab_size) # save the embedding matrix
    
    return args, task, model, criterion, embedding_weight, bpe