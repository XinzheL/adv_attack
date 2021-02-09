def translate():
    # build a Generator for translate
    generator = fairseq.sequence_generator.SequenceGenerator(
                task.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                # sampling=getattr(args, 'sampling', False),
                # sampling_topk=getattr(args, 'sampling_topk', -1),
                # sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                # diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                # diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    for samples in itr: 
        
        if torch.cuda.is_available() and not args.cpu:
            samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
            samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()

        with torch.no_grad():
            translations = generator.generate([model], samples) # keys: tokens,score,attention,alignment,positional_scores
        original_prediction = translations[0][0]['tokens']  # shape: [seq_len, ]
        

        # int -> string
        if bpe is not None:
            x = bpe.decode(task.source_dictionary.string(samples['net_input']['src_tokens'], None))
            y_true = bpe.decode(task.target_dictionary.string(samples['target']))
            original_output_tokens = bpe.decode(task.target_dictionary.string(original_prediction, None)) 
        else: 
            x = task.source_dictionary.string(samples['net_input']['src_tokens'], None)
            y_true = task.target_dictionary.string(samples['target'])
            original_output_tokens = task.target_dictionary.string(original_prediction, None)
        print('Input sample:', x, '\n Reference translation:', y_true, '\n Model translation: ', original_output_tokens)
def attack():


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

        # new prev_output_tokens: tensor([[    2,    74,   577, 11496]], device='cuda:0')
        samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)
        
        return samples, target_mask


    
    # criterion = fairseq.criterions.build_criterion(args, task) # default cross_entropy
    

    import json
    # for targeted flips to find antonyms to replace
    with open('adv_antonyms.json', 'r') as fp:
        antonym_candidates = json.load( fp)

    trainer = Trainer(args, task, model, adv_criterion)
    if len(antonym_candidates[i]) == 0:
        logging.debug('Ignore sample '+str(i)+ ' due to no anonyms found.')
        continue
    ##### 2. Now define adversarial output for attack
    # take the first antonym pair found by wordnet
    antonyms = antonym_candidates[samples['id']]
    if len(antonyms) > 0:
        original_output_token, desired_output_token = list(antonyms.keys())[0], list(antonyms.values())[0][0] # string
    
    else: # cannot find the antonyms
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


    ##### 3. Now begin attack
    assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure eos is always there    

    
    # 3.1 clear grads, compute new grads, and get candidate tokens ; shape: (T-1, k)
    input_gradients = get_input_gradients(trainer, samples, target_mask=target_mask)
    

    new_found_input_tokens = None
    batch_size =  all_attack_utils.check_bsz(samples, args, candidate_input_tokens, trainer, bpe) # so batch number ~= (T-1)*k / batch_size (less than due to invalid replacement)
    all_inference_samples, all_changed_positions = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe)
    #all_inference_samples_orig, all_changed_positions_orig = all_attack_utils.build_inference_samples_orig(samples, batch_size, args, candidate_input_tokens, trainer, bpe, )
    
    # 3.2 Test candidates
    print('Begin test candidates.')
    for j, inference_sample in enumerate(all_inference_samples):
        if j % 100 == 0:
            print(f"Already test {j} examples.")

        with torch.no_grad():
            predictions = generator.generate([model], inference_sample, prefix_tokens=None)

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
        
def invalid_input_bpe_segmentation(src_tokens, bpe, src_dict, cpu=False):
    """
    there are cases where making a BPE swap would cause the BPE segmentation to change.
    in other words, the input we are using would be invalid because we are using an old segmentation
    for these cases, we just skip those candidates   
       
    """
    string_input_tokens = bpe.decode(src_dict.string(src_tokens, None))
    retokenized_string_input_tokens = src_dict.encode_line(bpe.encode(string_input_tokens)).long().unsqueeze(dim=0)
    if torch.cuda.is_available() and not cpu:
        retokenized_string_input_tokens = retokenized_string_input_tokens.cuda()
    if len(retokenized_string_input_tokens[0]) != len(src_tokens) or \
        not torch.all(torch.eq(retokenized_string_input_tokens[0],src_tokens)):
        
        return True
    return False



# take samples (which is batch size 1) and repeat it batch_size times to do batched inference / loss calculation
# for all of the possible attack candidates
def build_inference_samples_orig(samples, batch_size, args, candidate_input_tokens, trainer, bpe, changed_positions=None, untouchable_token_blacklist=None, adversarial_token_blacklist=None, num_trigger_tokens=None):
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

def check_bsz(samples, args, candidate_input_tokens, trainer, bpe):
    """ check if 1 optimized candidate for T-1 pos
    """
    bsz = len(candidate_input_tokens)
    
    for pos, token_id in enumerate(candidate_input_tokens):
        src_tokens = deepcopy(samples['net_input']['src_tokens'])
        src_tokens[0][pos] = torch.LongTensor([token_id]).squeeze(0) 
        if invalid_input_bpe_segmentation(src_tokens, bpe, trainer.task.source_dictionary, trainer.args.cpu):
            bsz -= 1
            print(bsz)
    return bsz



def build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe):
    """ Prepare batches of samples for inference
    Args:
    samples () : (which is batch size 1) and repeat it `batch_size` times to do batched inference / loss calculation
    for all of the possible attack candidates ; `prev_output_tokens` is the right shift of `src_tokens` and starts with eos 
    """


    # copy and repeat the samples instead batch size elements
    samples_repeated_by_batch = deepcopy(samples)
    samples_repeated_by_batch['ntokens'] *= batch_size # e.g. 4 ->256
    samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1) # e.g. shape: (B:1, T_y_hat) -> (B:64, T_y_hat) 
    samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1) # e.g. shape: (B:1, T_y_hat) -> (B:64, T_y_hat) 
    samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1) # e.g. shape: (B:1, T_x) -> (B:64, T_x) 
    samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1) # e.g. shape: (1) -> (B: 64, 1)
    samples_repeated_by_batch['nsentences'] = batch_size 

    all_inference_samples = [] # stores a list of batches of candidates
    all_changed_positions = [] # stores all the changed_positions for each batch element

    
    current_batch_changed_position = []
    current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
    net_input = current_inference_samples['net_input']
    batch_idx = 0
          
    if len(candidate_input_tokens.shape) == 1:
        candidate_input_tokens = candidate_input_tokens.reshape(-1, 1)

    for pos in range(len(candidate_input_tokens)): # for all T-1 positions in the input 
        
        for token_id in candidate_input_tokens[pos]: # for all k candidates 
            
            # `current_inference_samples` src_tokens changed inplaced: change one token or not if invalid
            original_token = deepcopy(net_input['src_tokens'][batch_idx][pos]) 
            net_input['src_tokens'][batch_idx][pos] = torch.LongTensor([token_id]).squeeze(0) 
            # check invalid_input_bpe_segmentation
            if invalid_input_bpe_segmentation(net_input['src_tokens'][batch_idx], bpe, trainer.task.source_dictionary, trainer.args.cpu):
                # undo the token we replaced and move to the next candidate
                net_input['src_tokens'][batch_idx][pos] = original_token
                # continue to keep batch_idx unchanged so that next loop will change the same item in `current_inference_samples`
                continue

            # `current_batch_changed_position` 
            current_batch_changed_position.append(pos) 
            batch_idx += 1

            if batch_idx == batch_size:
                all_inference_samples.append(deepcopy(current_inference_samples))
                all_changed_positions.append(current_batch_changed_position)

                # update to build another batch during the loop
                current_batch_changed_position = []
                current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
                net_input = current_inference_samples['net_input']
                batch_idx = 0


    return all_inference_samples, all_changed_positions


