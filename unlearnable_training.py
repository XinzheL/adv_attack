
TRAIN_TYPE = 'normal' # 'error_min' # 'error_max' # 
output_dir = "checkpoints/bi_sst/lstm/"

# load training data 
READER_TYPE= None # 'pretrained' #
pretrained_model = None # 'bert-base-uncased' #
MODEL_TYPE=  'lstm' # 'finetuned_bert' #
EMBEDDING_TYPE = "w2v" # None # 

from allennlp.data.vocabulary import Vocabulary
if READER_TYPE == 'pretrained':
    vocab = Vocabulary.from_files('output/vocab')

label_ids = [0, 1] # I hope it equals to label in LabelField, TODO: re-train it for lstm
from utils.allennlp_data import load_sst_data
reader, train_data = load_sst_data('train', \
    READER_TYPE=READER_TYPE, \
    pretrained_model = pretrained_model)

_, dev_data = load_sst_data('dev', \
    READER_TYPE=READER_TYPE, \
    pretrained_model = pretrained_model)

from utils.allennlp_model import train_sst_model

if TRAIN_TYPE == 'normal':
    train_sst_model(output_dir, list(train_data), list(dev_data), \
        MODEL_TYPE=MODEL_TYPE, \
        EMBEDDING_TYPE = EMBEDDING_TYPE,  \
        pretrained_model = pretrained_model)
        

elif TRAIN_TYPE == 'error_max':

    from utils.universal_attack import UniversalAttack
    from allennlp.data.tokenizers import Token
    import pandas as pd
    noisy_train_data = None
    # load trigger tokens and prepend to the instances with correponding class
    for label in label_ids: 
        trigger_tokens = list(pd.read_csv(f"result_data/{MODEL_TYPE}_{str(label)}.csv")['triggers'])[-1].split('_')
        # TODO: change `trigger_tokens` to dict which contains all the classes so no need classmethods
        if noisy_train_data is None:
            
            noisy_train_data = UniversalAttack.prepend_batch(
                UniversalAttack.filter_instances( \
                    list(train_data), label_filter=label, vocab=vocab
                ), \
                trigger_tokens=[Token(token_txt) for token_txt in trigger_tokens], \
                vocab = vocab
            )
        else:
            noisy_train_data += UniversalAttack.prepend_batch(
                UniversalAttack.filter_instances( \
                    list(train_data), label_filter=label, vocab=vocab
                ), \
                trigger_tokens=[Token(token_txt) for token_txt in trigger_tokens], \
                vocab = vocab
            )


    # TODO: make noisy_train_data be `Generator`
    train_sst_model("unlearnabe_output/", noisy_train_data, dev_data, \
        MODEL_TYPE=MODEL_TYPE, \
        EMBEDDING_TYPE = EMBEDDING_TYPE,  \
        pretrained_model = pretrained_model)



elif TRAIN_TYPE == 'error_min':
    pass
