
### USER INPUT 
TRAIN_TYPE = 'error_min' # None # 'error_min' # 
MODEL_TYPE = 'finetuned_bert' #'lstm' #
num_epochs=3
bsz = 32
###

if TRAIN_TYPE is None:
    MODELS_DIR = f'checkpoints/bi_sst/'
else:
    MODELS_DIR = f'checkpoints/bi_sst_{TRAIN_TYPE}/'

output_dir = f"{MODELS_DIR}{MODEL_TYPE}/"

if MODEL_TYPE == 'finetuned_bert':
    READER_TYPE= 'pretrained' # None # 
    pretrained_model = 'bert-base-uncased' # None # 
    EMBEDDING_TYPE = None # "w2v" # 

elif MODEL_TYPE =='lstm' :
    READER_TYPE= None 
    pretrained_model = None 
    EMBEDDING_TYPE = "w2v" 

# load training data 
from allennlp.data.vocabulary import Vocabulary
if TRAIN_TYPE == 'error_max':
    if READER_TYPE == 'pretrained':
        vocab = Vocabulary.from_files(f'checkpoints/bi_sst/{MODEL_TYPE}/vocab/')
    else:
        vocab = Vocabulary.from_files(f'checkpoints/bi_sst/{MODEL_TYPE}/vocab/')

label_ids = [0, 1] # I hope it equals to label in LabelField, TODO: re-train it for lstm
from utils.allennlp_data import load_sst_data
reader, train_data = load_sst_data('train', \
    READER_TYPE=READER_TYPE, \
    pretrained_model = pretrained_model)

_, test_data = load_sst_data('test', \
    READER_TYPE=READER_TYPE, \
    pretrained_model = pretrained_model)

train_data, test_data = list(train_data), list(test_data)

# train model
from utils.allennlp_model import train_sst_model

        

if TRAIN_TYPE == 'error_max':

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
    train_data = noisy_train_data

if TRAIN_TYPE == 'error_min':
    pass
    

train_sst_model(output_dir, train_data, test_data, \
    MODEL_TYPE=MODEL_TYPE, \
    EMBEDDING_TYPE = EMBEDDING_TYPE,  \
    pretrained_model = pretrained_model, num_epochs=num_epochs, bsz = bsz,\
    TRAIN_TYPE=TRAIN_TYPE)

    
 

    

    
