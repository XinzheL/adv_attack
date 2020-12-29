# import fairseq
import torch
import fairseq
from fairseq.trainer import Trainer
import all_attack_utils


# input_args =['--arch', 'transformer_vaswani_wmt_en_de_big', # this would be used for setting model arch args
#              '--restore-file', 'wmt16.en-de.joined-dict.transformer/model.pt',
#              '--bpe', 'subword_nmt',
#              '--bpe-codes', 'wmt16.en-de.joined-dict.transformer/bpecodes',
#              '--interactive-attacks',
#              'wmt16.en-de.joined-dict.newstest2014/',]

             

# input_args =['--arch', 'transformer_vaswani_wmt_en_fr_big',
#              '--restore-file', 'wmt14.en-fr.joined-dict.transformer/model.pt',
#              '--bpe', 'subword_nmt',
#              '--bpe-codes', 'wmt14.en-fr.joined-dict.transformer/bpecodes',
#              '--interactive-attacks',
#              '--source-lang', 'en',  '--target-lang', 'fr',
#              'wmt14.en-fr.newstest2014',]            



input_args =['--arch', 'fconv_wmt_en_fr',
             '--restore-file', 'wmt14.en-fr.fconv-py/model.pt',
             '--bpe', 'subword_nmt',
             '--bpe-codes', 'wmt14.en-fr.fconv-py/bpecodes',
             '--interactive-attacks',
             '--source-lang', 'en',  '--target-lang', 'fr',
             'wmt14.en-fr.newstest2014',]


# input_args =['--arch', 'lstm_luong_wmt_en_de',
#              '--restore-file', 'wmt16.en-de.joined-dict.transformer/model.pt',
#              '--bpe', 'subword_nmt',
#              '--bpe-codes', 'wmt16.en-de.joined-dict.transformer/bpecodes',
#              '--interactive-attacks',
#              'wmt16.en-de.joined-dict.newstest2014/',]
# 

# setup 
args, task, model, criterion, embedding_weight, bpe = all_attack_utils.setup(input_args=input_args)    
trainer = Trainer(args, task, model, criterion)
itr = [None] * 100000  # a fake dataset to go through, overwritten when doing interactive attacks


# 
for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
    assert args.interactive_attacks # only interactive for now
    # get input string and vectorize it with src_dict
    # get user input and build samples
    samples = all_attack_utils.get_user_input(trainer, bpe)
    while samples is None:
        samples = all_attack_utils.get_user_input(trainer, bpe)

    if torch.cuda.is_available() and not trainer.args.cpu:
        samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
        samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()

    # translate
    # translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
    # samples: 
    # {'net_input': {'src_lengths': tensor([4], device='cuda:0'), 
    #               'src_tokens': tensor([[  322,   106, 19454,     2]]='cuda:0')}, 
    # 'ntokens': 4}
    # generator : fairseq.sequence_generator.SequenceGenerator
    # generator = task.build_generator(args)
    generator = fairseq.sequence_generator.SequenceGenerator(
                trainer.task.target_dictionary,
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

    with torch.no_grad():
        translations = generator.generate([trainer.get_model()], samples, prefix_tokens=None)
    original_prediction = translations[0][0]['tokens']  # keys: dict_keys(['tokens', 'score', 'attention', 'alignment', 'positional_scores'])  
    samples['target'] = original_prediction.unsqueeze(0)    # tensor([[   74,   577, 18057,  3730,     2]], device='cuda:0')
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:],
                                                            samples['target'][0][:-1]), dim=0).unsqueeze(dim=0) # tensor([[    2,    74,   577, 18057,  3730]], device='cuda:0')
    if torch.cuda.is_available() and not args.cpu:
        samples['target'] = samples['target'].cuda()
        samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()

    if args.interactive_attacks:        
        print('Current Translation ', bpe.decode(trainer.task.target_dictionary.string(original_prediction, None)))
        original_output_token = input('Enter the target token you want to flip ')
        desired_output_token = input('Enter the desired token you want to flip it to ')
        adversarial_token_blacklist_string = input('Enter an (optional) space seperated list of words you do not want the attack to insert ')
        untouchable_token_blacklist_string = input('Enter an (optional) space seperated list of source words you do not want to change ')

        # -1 strips off <eos> token
        original_output_token = trainer.task.target_dictionary.encode_line(bpe.encode(original_output_token)).long()[0:-1]
        desired_output_token = trainer.task.target_dictionary.encode_line(bpe.encode(desired_output_token)).long()[0:-1]
        print("Original Length of Target Token ", len(original_output_token), "Desired Output Token Length ", len(desired_output_token))

        # don't change any of these tokens in the input
        untouchable_token_blacklist = []
        if untouchable_token_blacklist_string is not None and untouchable_token_blacklist_string != '' and untouchable_token_blacklist_string != '\n':
            untouchable_token_blacklist_string = untouchable_token_blacklist_string.split(' ')
            for token in untouchable_token_blacklist_string:
                token = trainer.task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]
                untouchable_token_blacklist.extend(token)

        # don't insert any of these tokens (or their synonyms) into the source
        adversarial_token_blacklist = []
        adversarial_token_blacklist.extend(desired_output_token) # don't let the attack put these words in
        if adversarial_token_blacklist_string is not None and adversarial_token_blacklist_string != '' and adversarial_token_blacklist_string != '\n':
            adversarial_token_blacklist_string = adversarial_token_blacklist_string.split(' ')
            synonyms = set()
            for token in adversarial_token_blacklist_string:
                token = trainer.task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]
                if len(token) == 1:
                    adversarial_token_blacklist.append(token)
                    for syn in wordnet.synsets(bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(token), None))): # don't add any synonyms either
                        for l in syn.lemmas():
                            synonyms.add(l.name())
            for synonym in synonyms:
                synonym_bpe = trainer.task.source_dictionary.encode_line(bpe.encode(synonym)).long()[0:-1]
                untouchable_token_blacklist.extend(synonym_bpe)
    

    # find the position of the start and end of the original_output_token and replaces it with desired_output_token
# desired_output_token can be shorter, longer, or the same length as original_output_token
    def find_and_replace_target(samples, original_output_token, desired_output_token):
        print("Original:", samples['target'])
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
        print("After Replace:", samples['target'])
        return samples, target_mask

    from copy import deepcopy
    # overwrite target with user desired output
    # sample target: tensor([[   74,   577, 11496,     2]], device='cuda:0')
    # target mask: [0, 0, 1, 0]
    samples, target_mask = find_and_replace_target(samples, original_output_token, desired_output_token)
    original_samples = deepcopy(samples)
    original_target = deepcopy(samples['target'])

    new_found_input_tokens = None
    best_found_loss = 999999999999999
    samples = deepcopy(original_samples)

    for i in range(samples['ntokens'] * 3): # this many iters over a single example. Gradient attack will early stop
        assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad is always there        
        assert all(torch.eq(samples['target'].squeeze(0), original_target.squeeze(0))) # make sure target is never updated

        if new_found_input_tokens is not None:            
            print('\nFinal input', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
            print('Final output ', bpe.decode(trainer.task.target_dictionary.string(translations[0][0]['tokens'], None)))
            break

        # clear grads, compute new grads, and get candidate tokens
        candidate_input_tokens = all_attack_utils.get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=500, target_mask=target_mask)

        new_found_input_tokens = None
        batch_size = 64
        all_inference_samples, _ = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe, untouchable_token_blacklist=untouchable_token_blacklist, adversarial_token_blacklist=adversarial_token_blacklist)

        for inference_sample in all_inference_samples:
            predictions = trainer.task.inference_step(generator, [trainer.get_model()],
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
        if new_found_input_tokens is not None:
            samples['net_input']['src_tokens'] = new_found_input_tokens # updating samples doesn't matter because we are done
        else: # get losses and find the best one to keep making progress
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
