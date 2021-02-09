import torch
from fairseq import options, tasks
import fairseq



def load_task_and_model(args):

    # set up task
    task = tasks.setup_task(args)
    # build bpe: fairseq.data.encoders.subword_nmt_bpe.SubwordNMTBPE object
    bpe = fairseq.data.encoders.build_bpe(args)
    # load model : call `checkpoint_utils.load_model_ensemble(filename=args.path.split(':'))` to load all from file

    def load_model(state_path, task, args):
        from torch.serialization import default_restore_location

        # 1). loads a checkpoint to CPU
        # `default_restore_location()` finds the working device matching 'cpu' from `_package_registry` list according to priority
        state = torch.load(
            state_path, map_location=lambda s, l: default_restore_location(s, 'cpu'),
        )
        
        #state = checkpoint_utils._upgrade_state_dict(state) # upgrading for backward compatibility
        # 2). build a model (no care for parameters) : `models.build_model(arch)`
        # here model arch is saved in the state checkpoint model.pt file not from user input_args
        model_arch = fairseq.models.ARCH_MODEL_REGISTRY[state['args'].arch] # ?for lstm state['args'] is None
        # To-Do: build model architecture using state['args'] for consistency
        model = model_arch.build_model(args, task)
        
        # 3). load checkpoint parameters into `model`
        model.load_state_dict(state['model'])
        return model
    
    model = load_model(args.path, task, args) # need dictionaries from task
    # load model to gpu
    if torch.cuda.is_available() and not args.cpu:
        assert torch.cuda.device_count() == 1 # only works on 1 GPU for now
        torch.cuda.set_device(0)
        model.cuda()
    # model.make_generation_fast_(beamable_mm_beam_size=args.beam, need_attn=False)
    return task, model, bpe


def get_user_input(task, model, bpe=None):
    user_input = input('Enter the input sentence that you to attack: ')
    if user_input.strip() == '':
        print("You entered a blank token, try again")
        return None

    # as in LanguagePairDataset() collate([{'id': 605, 'source': tensor([9860,  363,    2]), 'target': tensor([7022, 6838,    2])}], pad_idx=1, eos_idx=2)
    return process_sample(user_input, 
                          task.source_dictionary,
                          bpe,
                          model.encoder.embed_tokens.weight.shape[0])



def process_sample(sample, dictionary, bpe, vocab_size):
    # tokenize/vectorize input and get lengths 
    # 'i am sad'
    if bpe is not None:
        sample = bpe.encode(sample)
    # tensor([[  322,   106, 19454,     2]]) ps. 2 is `eos_index`
    sample_tokenized = dictionary.encode_line(sample).long().unsqueeze(dim=0)

    # check if the user input a token with is an UNK
    for token in sample_tokenized[0]:
        if torch.eq(token, vocab_size) or torch.gt(token, vocab_size): # >= max vocab size
            print('You entered an UNK token for your model, please try again. This usually occurs when (1) you entered '
                ' unicode or other strange symbols, (2) your model uses a lowercased dataset but you entered uppercase, or '
                ' (3) your model is expecting apostrophies as &apos; and quotes as &quot;')
            return None
    
    length_user_input = torch.LongTensor([len(sample_tokenized[0])]) # [4]

    # build samples which is input to the model
    # ?? use of `ntokens`, same as `src_lengths`
    return {'net_input': 
                {'src_tokens': sample_tokenized_bpe, 'src_lengths': length_user_input}, 
            'ntokens': len(sample_tokenized_bpe[0])}