# import fairseq
import torch
import fairseq
from fairseq.utils import resolve_max_positions
from fairseq.trainer import Trainer
import all_attack_utils
from copy import deepcopy

import logging
logging.basicConfig(filename='attack_info.txt', level=logging.DEBUG)

input_args =['--arch', 'fconv_wmt_en_fr',
             '--restore-file', 'wmt14.en-fr.fconv-py/model.pt',
             '--bpe', 'subword_nmt',
             '--bpe-codes', 'wmt14.en-fr.fconv-py/bpecodes',
             '--source-lang', 'en',  '--target-lang', 'fr',
             'wmt14.en-fr.newstest2014',]

def define_blacklist_string():
    # don't change any of these tokens in the input
    untouchable_token_blacklist_string = input('Enter an (optional) space seperated list of source words you do not want to change ')
    untouchable_token_blacklist = []
    if untouchable_token_blacklist_string is not None and untouchable_token_blacklist_string != '' and untouchable_token_blacklist_string != '\n':
        untouchable_token_blacklist_string = untouchable_token_blacklist_string.split(' ')
        for token in untouchable_token_blacklist_string:
            token = task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]
            untouchable_token_blacklist.extend(token)

    # don't insert any of these tokens (or their synonyms) into the source
    adversarial_token_blacklist_string = input('Enter an (optional) space seperated list of words you do not want the attack to insert ')
    adversarial_token_blacklist = []
    adversarial_token_blacklist.extend(desired_output_token) # don't let the attack put these words in
    if adversarial_token_blacklist_string is not None and adversarial_token_blacklist_string != '' and adversarial_token_blacklist_string != '\n':
        adversarial_token_blacklist_string = adversarial_token_blacklist_string.split(' ')
        synonyms = set()
        for token in adversarial_token_blacklist_string:
            token = task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]
            if len(token) == 1:
                adversarial_token_blacklist.append(token)
                for syn in wordnet.synsets(bpe.decode(task.source_dictionary.string(torch.LongTensor(token), None))): # don't add any synonyms either
                    for l in syn.lemmas():
                        synonyms.add(l.name())
        for synonym in synonyms:
            synonym_bpe = task.source_dictionary.encode_line(bpe.encode(synonym)).long()[0:-1]
            untouchable_token_blacklist.extend(synonym_bpe)

    return untouchable_token_blacklist, adversarial_token_blacklist
    
     


 # find the position of the start and end of the original_output_token and replaces it with desired_output_token
# desired_output_token can be shorter, longer, or the same length as original_output_token
def find_and_replace_target(samples, original_output_token, desired_output_token):
    target_mask = []
    start_pos = None
    end_pos = None
    for idx, current_token in enumerate(samples['target'].cpu()[0]): # tensor([   74,   577, 18057,  3730,     2])
        if current_token == original_output_token[0]: # TODO, the logic here will fail when a BPE id is repeated
            start_pos = idx
        if current_token == original_output_token[-1]:
            end_pos = idx
    if start_pos is None or end_pos is None:
        exit('find and replace target failed to find token')

    
    # old target: tensor([[   74,   577, 18057,  3730,     2]], device='cuda:0')
    # new target: tensor([[   74,   577, 11496,     2]], device='cuda:0')
    last_tokens_of_target = deepcopy(samples['target'][0][end_pos+1:])
    new_start = torch.cat((samples['target'][0][0:start_pos], desired_output_token.cuda()), dim=0)
    new_target = torch.cat((new_start, last_tokens_of_target), dim=0)
    target_mask = [0] * start_pos + [1] * len(desired_output_token) + [0] * (len(new_target) - len(desired_output_token) - start_pos)
    samples['target'] = new_target.unsqueeze(0)

    # tensor([[    2,    74,   577, 11496]], device='cuda:0')
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)
    
    return samples, target_mask



def generate_adv_ouput(y_hat):
    """ generate the antonym to replace for one ouput token
    """
    from nltk.corpus import wordnet
    for token in y_hat.split(' '):
        original_output_token = token.strip().lower() # input('Enter the target token you want to flip ') 
        
        
        syns = wordnet.synsets(original_output_token, lang='fra')  # english since they are just multilingual lemmas, i.e. synsets are language agnostic
        if not len(syns):
            continue
        original_output_token_lemmas = syns[0].lemmas()
        antonyms_lemmas = original_output_token_lemmas[0].antonyms() 
        if not len(antonyms_lemmas): # No antonyms found
            continue
        
        # get antonym in French
        antonyms_fra = antonyms_lemmas[0].synset().lemma_names('fra')
        # if not len(antonyms_fra): # no corresponding French found
        #     continue

        desired_output_token = antonyms_fra[0]  # input('Enter the desired token you want to flip it to ')
        return original_output_token,  desired_output_token
    return None, None

# setup task, model, loss function, bpe
args, task, model, criterion, embedding_weight, bpe = all_attack_utils.setup(input_args=input_args)    
trainer = Trainer(args, task, model, criterion)

# setup and load valid data
args.valid_subset = 'test'
for valid_sub_split in args.valid_subset.split(','): 
    print('args.valid_subset: ' + args.valid_subset)
    task.load_dataset(valid_sub_split, combine=False, epoch=0)  # load valid data from `args.data` into datasets[split] if not provide via cmd prompt

# return fairseq.iterators.EpochBatchIterator: a batched iterator over the given dataset split
itr = task.get_batch_iterator(dataset=task.dataset(args.valid_subset),
                                max_tokens=args.max_tokens_valid,
                                max_sentences=args.max_sentences_valid,
                                max_positions=resolve_max_positions(task.max_positions(), (task.max_positions()[0]-2, task.max_positions()[1]-2), ), # task max pos and model max pos
                                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                                required_batch_size_multiple=args.required_batch_size_multiple,
                                seed=args.seed,
                                num_shards=args.distributed_world_size,
                                shard_id=args.distributed_rank,
                                num_workers=args.num_workers,) # EpochBatchIterator
itr = itr.next_epoch_itr(shuffle=False) # CountingIterator


num_examples = len(itr)
adv_src = [None] * num_examples
out = [None] * num_examples
adv_out = [None] * num_examples

# build a Generator for translate
    # translations = task.inference_step(generator, [model], samples)
    # samples: 
    # {'net_input': {'src_lengths': tensor([4], device='cuda:0'), 
    #               'src_tokens': tensor([[  322,   106, 19454,     2]]='cuda:0')}, 
    # 'ntokens': 4}
    # generator : fairseq.sequence_generator.SequenceGenerator
    # generator = task.build_generator(args)
generator = fairseq.sequence_generator.SequenceGenerator(
            task.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
        )


# samples e.g. {'id': tensor([605]), 'net_input': {'prev_output_tokens': tensor([[   2, 7022, 6838]]), 'src_lengths': tensor([3]), 'src_tokens': tensor([[9860,  363,    2]])}, 'nsentences': 1, 'ntokens': 3, 'target': tensor([[7022, 6838,    2]])}
for i, samples in enumerate(itr): # itr: fairseq.data.iterators.CountingIterator
    
    if torch.cuda.is_available() and not args.cpu:
        samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
        samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()


    with torch.no_grad():
        translations = generator.generate([model], samples, prefix_tokens=None)
    original_prediction = translations[0][0]['tokens']  # keys: dict_keys(['tokens', 'score', 'attention', 'alignment', 'positional_scores'])  
    samples['target'] = original_prediction.unsqueeze(0)    # tensor([[   74,   577, 18057,  3730,     2]], device='cuda:0')
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:],
                                                            samples['target'][0][:-1]), dim=0).unsqueeze(dim=0) # tensor([[    2,    74,   577, 18057,  3730]], device='cuda:0')
    if torch.cuda.is_available() and not args.cpu:
        samples['target'] = samples['target'].cuda()
        samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()

    x = bpe.decode(task.source_dictionary.string(samples['net_input']['src_tokens'], None))
    y_true = bpe.decode(task.target_dictionary.string(samples['target']))
    original_output_tokens = bpe.decode(task.target_dictionary.string(original_prediction, None)) # int -> string
    
    # from bleu_score import sentence_bleu
    # from chrf_score import sentence_chrf
    # print('BLEU score ', sentence_bleu([y_true], original_output_tokens))
    # print('CHRF score ', sentence_chrf([y_true], original_output_tokens))
    
    ##### Now define adversarial ouput for attack
    # take the random token from the sample input    
    original_output_token, desired_output_token = generate_adv_ouput(original_output_tokens) # string
    if original_output_token is None: # cannot find the antonyms
        continue

    # encode ; -1 strips off <eos> token
    original_output_token = task.target_dictionary.encode_line(bpe.encode(original_output_token)).long()[0:-1]
    desired_output_token = task.target_dictionary.encode_line(bpe.encode(desired_output_token)).long()[0:-1]

    # overwrite target with user desired output
    # sample target: tensor([[   74,   577, 11496,     2]], device='cuda:0')
    # target mask: [0, 0, 1, 0]
    samples, target_mask = find_and_replace_target(samples, original_output_token, desired_output_token)
    # logging.debug('Original Input : ' + x)
    # logging.debug('Original Translation ' + original_prediction.__str__() + '   ' + original_output_tokens)
    # logging.debug("Original Target Token " + original_output_token.__str__() +  "   Desired Output Token " + " " + desired_output_token.__str__())
    # logging.debug('Desired Translation ' + samples['target'].__str__() + "   " + bpe.decode(task.target_dictionary.string(samples['target'])))
    

    # define blacklist tokens
    untouchable_token_blacklist, adversarial_token_blacklist = define_blacklist_string()
  

    ##### Now begin attack
    assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure eos is always there    

    # clear grads, compute new grads, and get candidate tokens ; shape: (T-1, k)
    candidate_input_tokens = all_attack_utils.get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=500, target_mask=target_mask)

    new_found_input_tokens = None
    batch_size = 64 # so batch number ~= (T-1)*k / batch_size (less than due to invalid replacement)
    all_inference_samples, all_changed_positions = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe, untouchable_token_blacklist=untouchable_token_blacklist, adversarial_token_blacklist=adversarial_token_blacklist)
    #all_inference_samples_orig, all_changed_positions_orig = all_attack_utils.build_inference_samples_orig(samples, batch_size, args, candidate_input_tokens, trainer, bpe, untouchable_token_blacklist=untouchable_token_blacklist, adversarial_token_blacklist=adversarial_token_blacklist)
    
    for inference_sample in all_inference_samples:
        # with torch.no_grad():
        #     predictions generator.generate([model], inference_sample, prefix_tokens=None)
        predictions = task.inference_step(generator, [model],
            inference_sample) # batched inference
        for prediction_indx, prediction in enumerate(predictions): # for all predictions
            prediction = prediction[0]['tokens'].cpu()
            # if prediction is the same, then save input
            desired_output_token_appeared = False
            original_output_token_present = False

            if all(token in prediction for token in desired_output_token): # we want the desired_output_token to be present
                desired_output_token_appeared = True
            if any(token in prediction for token in original_output_token):  # and we want the original output_token to be gone
                original_output_token_present = True
            if desired_output_token_appeared and not original_output_token_present:
                new_found_input_tokens = deepcopy(inference_sample['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                break
            if new_found_input_tokens is not None:
                break
        if new_found_input_tokens is not None:
            break
    if new_found_input_tokens is not None: # tensor([[10506,  5764->23434,  1734,     2]]
        samples['net_input']['src_tokens'] = new_found_input_tokens # updating samples doesn't matter because we are done           
        
        translations = task.inference_step(generator, [model], samples)
        # logging.debug('\nFinal Input' + bpe.decode(task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
        # logging.debug('Final Translation ' + bpe.decode(task.target_dictionary.string(translations[0][0]['tokens'], None)))    
        
        # save original output, adv input and output for evaluation
        idx = samples['id'].item() # index for unshuffle test data
        out[idx] = original_output_tokens
        adv_out[idx] = bpe.decode(task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0]))
        adv_src[idx] = bpe.decode(task.target_dictionary.string(translations[0][0]['tokens'], None))
        
    else: # get losses and find the best one to keep making progress
        best_found_loss = 999999999999999
        current_best_found_loss = 99999999
        current_best_found_tokens = None
        for inference_sample in all_inference_samples:
            _, losses = all_attack_utils.get_loss_and_input_grad(trainer, inference_sample, target_mask, no_backwards=True, reduce_loss=False)
            losses = losses.reshape(batch_size, samples['target'].shape[1]) # unflatten losses
            losses = torch.sum(losses, dim=1) # total loss. Note that for each entry of the batch, all entries are 0 except one.
            for loss_indx, loss in enumerate(losses):
                if loss < current_best_found_loss:
                    current_best_found_loss = loss
                    current_best_found_tokens = inference_sample['net_input']['src_tokens'][loss_indx].unsqueeze(0)

        if current_best_found_loss < best_found_loss: # update best tokens
            best_found_loss = current_best_found_loss
            samples['net_input']['src_tokens'] = current_best_found_tokens

        # gradient is deterministic, so if it didnt flip another then its never going to
        else:
            break

        
# Final input Need OFFgrowing
# Final output  Besoin d&apos; accroître


# Current Sentence  It was normal .
# Current Translation  tensor([[ 491,  135, 5467,    7,    2]], device='cuda:0') -> 'C&apos; était normal .'
# After Replace: tensor([[  491,   135, 12790,  2753,     7,     2]], device='cuda:0') -> 'C&apos; était anormal .'