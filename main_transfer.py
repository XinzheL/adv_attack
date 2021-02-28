

from utils.allennlp_model import load_sst_model, load_all_trained_models
from utils.allennlp_predictor import AttackPredictorForBiClassification
from utils.allennlp_data import load_sst_data
from allennlp.interpret.attackers import Hotflip
from utils.universal_attack import UniversalAttack
import os
import pandas as pd
from copy import deepcopy
from allennlp.data.tokenizers import Token

import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_triggers(triggers_log, MODELS, test_data, label_filter):
    """
    Args:
        triggers List[str] : a sequence of tokens to be prepended
        MODELS [Dict] : each pair stores model names as key and (loaded model, vocabulary) as value
    """
    accs = {}
    loss_lst = {}
    
    for triggers in triggers_log: 
        triggers = [Token(trigger) for trigger in triggers.split('_')]

        for M in MODELS.keys():
            model, vocab = MODELS[M]
            #Evaluate with test data appended by triggers
            noisy_test_data = UniversalAttack.prepend_batch(
                UniversalAttack.filter_instances( \
                    list(test_data), label_filter=label_filter, vocab=vocab
                ), \
                trigger_tokens=triggers, \
                vocab = vocab
            )

            acc, loss = UniversalAttack.evaluate_instances_cls(noisy_test_data, model, vocab, cuda_device=0)
            if M not in accs.keys():
                accs[M] = [acc]
                loss_lst[M] = [loss]
            else:
                accs[M].append(acc)
                loss_lst[M].append(loss)
    
    return accs, loss_lst

if __name__ == "__main__":
    MODELS_DIR = 'checkpoints/bi_sst/'
    sst_granularity = 2
    MODEL_TYPES = ['lstm' , 'lstm_w2v', 'cnn', 'cnn_w2v']

    # test triggers for ones in `MODELS_TO_CHOOSE` with chosen label
    # each has a `log_trigger_tokens`
    MODELS_TO_CHOOSE = ['lstm', 'lstm_w2v', 'cnn', 'cnn_w2v' ] 
    label = 0


    if not os.path.isfile( f'result_data/transfer_acc{str(label)}.csv'):
        # MODELS to test on
        MODELS = load_all_trained_models(MODELS_DIR, MODEL_TYPES = MODEL_TYPES)

        # test data
        _, test_data = load_sst_data('test', MODEL_TYPE=None, granularity = str(sst_granularity)+'-class')
        test_data = list(test_data)

        
        # triggers to test
        accs_data = None
        loss_data = None


        for i, MODEL in enumerate(MODELS_TO_CHOOSE):
            result_df = pd.read_csv(f'result_data/{MODEL}_{label}.csv')[['iteration', 'triggers', 'accuracy','loss']]
            log_trigger_tokens = list(result_df['triggers'])
            log_trigger_tokens = [log_trigger_tokens[-1]] # only evaluate on the last one

            if accs_data is None:
                accs_data, loss_data = evaluate_triggers(log_trigger_tokens, MODELS, deepcopy(test_data), label) # this is one row for one model model-generating triggers
            else:
                acc, loss = evaluate_triggers(log_trigger_tokens, MODELS, deepcopy(test_data), label)
                for k in acc.keys():
                    accs_data[k].append(acc[k][0])
                    loss_data[k].append(loss[k][0])
            

        df_acc = pd.DataFrame.from_dict(accs_data, orient='index', columns=MODEL_TYPES)
        df_loss = pd.DataFrame.from_dict(loss_data, orient='index', columns=MODEL_TYPES)
        df_acc.to_csv(f'result_data/transfer_acc{str(label)}.csv')
        df_loss.to_csv(f'result_data/transfer_loss{str(label)}.csv')
    else: 
        df_acc = pd.read_csv(f'result_data/transfer_acc{str(label)}.csv')
        df_loss =pd.read_csv(f'result_data/transfer_loss{str(label)}.csv')
        figure = plt.figure(figsize=(16, 6))
        axes = figure.subplots(2, 1)
        sns.heatmap(df_acc, ax=axes[0])
        sns.heatmap(df_loss, ax=axes[1])
        plt.show()