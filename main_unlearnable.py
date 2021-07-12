
def train(MODEL_TYPE, num_epochs=3, bsz = 32):
    

    # load training data 
    dataloader_train = load_data('bi_sst', 'train', MODEL_TYPE=None, distributed=False)
    train_data = list(dataloader_train.iter_instances())
    dataloader_dev = load_data('bi_sst', 'dev', MODEL_TYPE=None, distributed=False)
    dev_data = list(dataloader_dev.iter_instances())
    from allennlp.data.vocabulary import Vocabulary
    vocab = Vocabulary.from_instances(train_data)

    
    # train model
    LABELS = [0, 1]
    MODELS_DIR = f'checkpoints/bi_sst/'
    output_dir = f"{MODELS_DIR}{MODEL_TYPE}/"
    EMBEDDING_TYPE = "w2v"

    from utils.allennlp_model import train_sst_model
    from utils.universal_attack import UniversalAttack
    from allennlp.data.tokenizers import Token
    import pandas as pd

    label_ids = [0, 1] 
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


    train_sst_model(output_dir, train_data, test_data, MODEL_TYPE, \
        EMBEDDING_TYPE = EMBEDDING_TYPE, num_epochs=num_epochs, bsz = bsz,\
        TRAIN_TYPE=TRAIN_TYPE, LABELS=LABELS)




 
if __name__ == "__main__":
    TRAIN_TYPES = [None, 'error_max', 'error_min' ]
    MODEL_TYPES = ['distilbert-base-cased',  'distilroberta-base', 'bert-base-cased', 'roberta-base',  ] 
    for MODEL_TYPE in MODEL_TYPES:
        train(TRAIN_TYPE = None, MODEL_TYPE=MODEL_TYPE, \
            num_epochs=3, bsz = 32)
    

    

    
