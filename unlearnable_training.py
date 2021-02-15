
CHOOSE_MODEL = 'finetuned_bert' 


# load training data 


READER_TYPE='pretrained'
pretrained_model = 'bert-base-uncased'
MODEL_TYPE=  'finetuned_bert'
EMBEDDING_TYPE = None

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

from utils.universal_attack import UniversalAttack
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
            trigger_tokens=trigger_tokens, \
            vocab = vocab
        )
    else:
        noisy_train_data += UniversalAttack.prepend_batch(
            UniversalAttack.filter_instances( \
                list(train_data), label_filter=label, vocab=vocab
            ), \
            trigger_tokens=trigger_tokens, \
            vocab = vocab
        )

from utils.allennlp_model import train_sst_model
train_sst_model("output/", noisy_train_data, dev_data, \
    MODEL_TYPE=MODEL_TYPE, \
    EMBEDDING_TYPE = EMBEDDING_TYPE,  \
    pretrained_model = pretrained_model)




